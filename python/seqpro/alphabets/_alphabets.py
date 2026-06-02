from __future__ import annotations

from types import MethodType
from typing import Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from .._numba import (
    _nb_find_stop_ends,
    _pack_codon_index,
    gufunc_complement_bytes,
    gufunc_translate,
    gufunc_translate_lut,
    ufunc_comp_dna,
)
from .._utils import SeqType, StrSeqType, array_slice, cast_seqs, check_axes, is_dtype
from ..rag import Ragged, is_rag_dtype

OnUnknown = Literal["pad", "collapse", "shorten", "error"]
"""Policy for codons whose bytes aren't in ``{A, C, G, T, a, c, g, t}``.

* ``"pad"`` — emit ``unknown_marker`` once per unknown codon (1 marker per AA
  position). The shape of the output is preserved.
* ``"collapse"`` — emit a single ``unknown_marker`` per consecutive run of
  unknown codons (so a 3-AA deletion collapses to one marker). The output is
  shorter than the input, only available for Ragged input.
* ``"shorten"`` — drop unknown codons entirely; the output keeps only the
  successfully translated codons. The output is shorter than the input, only
  available for Ragged input.
* ``"error"`` — raise :class:`ValueError` naming the first codon position
  whose bytes were non-canonical (intended default for arbitrary callers).
"""

_ON_UNKNOWN_VALUES: tuple[OnUnknown, ...] = ("pad", "collapse", "shorten", "error")


def _validate_on_unknown(on_unknown: str) -> OnUnknown:
    """Return ``on_unknown`` unchanged if it's a recognised policy."""
    if on_unknown not in _ON_UNKNOWN_VALUES:
        raise ValueError(
            f"on_unknown must be one of {_ON_UNKNOWN_VALUES!r}; got {on_unknown!r}."
        )
    return cast(OnUnknown, on_unknown)


def _validate_unknown_marker(unknown_marker: str) -> np.uint8:
    """Validate that ``unknown_marker`` is a single ASCII byte and return its ord."""
    if not isinstance(unknown_marker, str):
        raise ValueError(
            f"unknown_marker must be a single-character str; got {type(unknown_marker).__name__}."
        )
    if len(unknown_marker) != 1:
        raise ValueError(
            f"unknown_marker must be exactly one character; got {unknown_marker!r} "
            f"(length {len(unknown_marker)})."
        )
    code = ord(unknown_marker)
    if code > 0xFF:
        raise ValueError(
            f"unknown_marker must be a single ASCII / latin-1 byte; got {unknown_marker!r}."
        )
    return np.uint8(code)


def _codon_unknown_mask(
    codons: NDArray[np.uint8],
    valid_upper_bytes: NDArray[np.uint8],
) -> NDArray[np.bool_]:
    """Return a boolean array marking codons that contain a non-canonical byte.

    Operates on the trailing (codon) axis of ``codons`` (shape ``(..., k)``).
    A codon is unknown if any of its bytes, after upper-casing via ``& 0xDF``,
    isn't in ``valid_upper_bytes``. Vectorised in numpy so it's cheap to call
    from the Python wrapper.

    Parameters
    ----------
    codons
        Codon bytes; the codon axis is the last one.
    valid_upper_bytes
        The alphabet's nucleotide bytes upper-cased via ``& 0xDF``. For the
        standard DNA alphabet this is ``ord({A, C, G, T})``; for a non-standard
        alphabet (e.g. one containing ``U``) it's the upper-cased
        ``_valid_nuc_bytes`` of that alphabet.
    """
    upper = codons & 0xDF
    canonical = np.isin(upper, valid_upper_bytes)
    return ~canonical.all(axis=-1)


def _shrink_ragged_unknowns(
    translated_flat: NDArray[np.bytes_],
    new_offsets: NDArray[np.int64],
    unknown_mask: NDArray[np.bool_],
    on_unknown: OnUnknown,
) -> tuple[NDArray[np.bytes_], NDArray[np.int64]]:
    """Per-sequence shrink of the flat translated AA buffer for collapse / shorten.

    ``translated_flat`` has leading shape ``(num_codons, *trailing)`` (the
    trailing axis is empty for bytes Ragged input and is the nucleotide-alphabet
    axis for OHE Ragged input). ``new_offsets`` is a 1-D ``(n+1,)`` array of
    codon-axis offsets in ``translated_flat``. ``unknown_mask`` is a 1-D
    ``(num_codons,)`` boolean array marking which codons were non-canonical.

    Returns a new ``(translated_flat, new_offsets)`` pair where every sequence
    has had its unknown codons collapsed or dropped per ``on_unknown``. Offsets
    remain monotonic and ``(n+1,)``-shaped, so the caller can hand them straight
    to ``Ragged.from_offsets``.
    """
    n_seq = len(new_offsets) - 1
    keep_parts: list[NDArray[np.bool_]] = []
    new_lengths = np.empty(n_seq, dtype=np.int64)
    for i in range(n_seq):
        start = int(new_offsets[i])
        end = int(new_offsets[i + 1])
        if end == start:
            keep_parts.append(np.zeros(0, dtype=np.bool_))
            new_lengths[i] = 0
            continue
        seq_mask = unknown_mask[start:end]
        if on_unknown == "shorten":
            keep = ~seq_mask
        else:  # collapse
            keep = np.ones(end - start, dtype=np.bool_)
            # Drop an unknown codon iff its previous (in-sequence) codon is
            # also unknown. The first codon of a sequence is always kept.
            keep[1:] = ~(seq_mask[1:] & seq_mask[:-1])
        keep_parts.append(keep)
        new_lengths[i] = int(keep.sum())

    keep_all = np.concatenate(keep_parts) if keep_parts else np.zeros(0, dtype=np.bool_)
    new_translated = translated_flat[keep_all]
    new_offsets_out = np.empty(n_seq + 1, dtype=np.int64)
    new_offsets_out[0] = 0
    np.cumsum(new_lengths, out=new_offsets_out[1:])
    return new_translated, new_offsets_out


class NucleotideAlphabet:
    alphabet: str
    """Alphabet excluding ambiguous characters e.g. "N" for DNA."""
    complement: str
    array: NDArray[np.bytes_]
    complement_map: dict[str, str]
    complement_map_bytes: dict[bytes, bytes]
    str_comp_table: dict[int, str]
    bytes_comp_table: bytes
    bytes_comp_array: NDArray[np.bytes_]

    def __init__(self, alphabet: str, complement: str) -> None:
        """Parse and validate sequence alphabets.

        Nucleic acid alphabets must be complemented by being reversed (without the
        unknown character). For example, `reverse(ACGT) = complement(ACGT) = TGCA`.

        Parameters
        ----------
        alphabet
            For example, DNA could be 'ACGT'.
        complement
            Complement of the alphabet, to continue the example this would be 'TGCA'.
        """
        self._validate(alphabet, complement)
        self.alphabet = alphabet
        self.complement = complement
        self.array = cast(
            NDArray[np.bytes_], np.frombuffer(self.alphabet.encode("ascii"), "|S1")
        )
        self.complement_map = dict(zip(list(self.alphabet), list(self.complement)))
        self.complement_map_bytes = {
            k.encode("ascii"): v.encode("ascii") for k, v in self.complement_map.items()
        }
        self.str_comp_table = str.maketrans(self.complement_map)
        self.bytes_comp_table = bytes.maketrans(
            self.alphabet.encode("ascii"), self.complement.encode("ascii")
        )
        self.bytes_comp_array = np.frombuffer(self.bytes_comp_table, "S1")

    def __len__(self):
        return len(self.alphabet)

    def _validate(self, alphabet: str, complement: str):
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet has repeated characters.")

        if len(set(complement)) != len(complement):
            raise ValueError("Complement has repeated characters.")

        for maybe_comp, comp in zip(alphabet[::-1], complement):
            if maybe_comp != comp:
                raise ValueError("Reverse of alphabet does not yield the complement.")

    def ohe(self, seqs: StrSeqType) -> NDArray[np.uint8]:
        """One hot encode a nucleotide sequence."""
        from .._encoders import ohe as _ohe

        return _ohe(seqs, self)

    def decode_ohe(
        self,
        seqs: NDArray[np.uint8],
        ohe_axis: int,
        unknown_char: str = "N",
    ) -> NDArray[np.bytes_]:
        """Convert an OHE array to an S1 byte array."""
        from .._encoders import decode_ohe as _decode_ohe

        return _decode_ohe(seqs, ohe_axis, self, unknown_char)

    @overload
    def tokenize(
        self,
        seqs: StrSeqType,
        token_map: dict[str, int],
        unknown_token: int,
        out: NDArray[np.int32] | None = None,
    ) -> NDArray[np.int32]: ...
    @overload
    def tokenize(
        self,
        seqs: Ragged[np.bytes_],
        token_map: dict[str, int],
        unknown_token: int,
        out: None = None,
    ) -> Ragged[np.int32]: ...
    def tokenize(self, seqs, token_map, unknown_token, out=None):
        """Tokenize sequences using the given token map. Delegates to sp.tokenize."""
        from .._encoders import tokenize as _tokenize

        return _tokenize(seqs, token_map, unknown_token, out)

    @overload
    def decode_tokens(
        self,
        seqs: NDArray[np.int32],
        token_map: dict[str, int],
        unknown_char: str = "N",
    ) -> NDArray[np.bytes_]: ...
    @overload
    def decode_tokens(
        self,
        seqs: Ragged[np.int32],
        token_map: dict[str, int],
        unknown_char: str = "N",
    ) -> Ragged[np.bytes_]: ...
    def decode_tokens(self, seqs, token_map, unknown_char="N"):
        """Decode token IDs back to sequences. Delegates to sp.decode_tokens."""
        from .._encoders import decode_tokens as _decode_tokens

        return _decode_tokens(seqs, token_map, unknown_char)

    def _complement_bytes(
        self, byte_arr: NDArray[np.bytes_], out: NDArray[np.bytes_] | None = None
    ) -> NDArray[np.bytes_]:
        if out is None:
            _out = out
        else:
            _out = out.view(np.uint8)
        _out = gufunc_complement_bytes(
            byte_arr.view(np.uint8), self.bytes_comp_array.view(np.uint8), _out
        )
        return _out.view("S1")

    def _rev_comp_byte(
        self,
        byte_arr: NDArray[np.bytes_],
        length_axis: int,
        out: NDArray[np.bytes_] | None = None,
    ) -> NDArray[np.bytes_]:
        out = self._complement_bytes(byte_arr, out)
        return np.flip(out, length_axis)

    @overload
    def reverse_complement(
        self,
        seqs: StrSeqType,
        length_axis: int | None = None,
        ohe_axis: int | None = None,
        out: NDArray[np.bytes_] | None = None,
    ) -> NDArray[np.bytes_]: ...
    @overload
    def reverse_complement(
        self,
        seqs: NDArray[np.uint8],
        length_axis: int | None = None,
        ohe_axis: int | None = None,
        out: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]: ...
    @overload
    def reverse_complement(
        self,
        seqs: SeqType,
        length_axis: int | None = None,
        ohe_axis: int | None = None,
        out: NDArray[np.bytes_ | np.uint8] | None = None,
    ) -> NDArray[np.bytes_ | np.uint8]: ...
    def reverse_complement(
        self,
        seqs: SeqType,
        length_axis: int | None = None,
        ohe_axis: int | None = None,
        out: NDArray[np.bytes_ | np.uint8] | None = None,
    ) -> NDArray[np.bytes_ | np.uint8]:
        """Reverse complement a sequence.

        Parameters
        ----------
        seqs
        length_axis
            Length axis, by default None
        ohe_axis
            One hot encoding axis, by default None

        Returns
        -------
        NDArray[np.bytes_ | np.uint8]
            Reverse-complemented sequences as S1 bytes or uint8 for OHE input.
        """
        check_axes(seqs, length_axis, ohe_axis)

        seqs_ = cast_seqs(seqs)

        if is_dtype(seqs_, np.bytes_):
            assert out is None or is_dtype(out, np.bytes_)
            if length_axis is None:
                length_axis = -1
            return self._rev_comp_byte(seqs_, length_axis, out)
        elif is_dtype(seqs_, np.uint8):  # OHE
            assert length_axis is not None
            assert ohe_axis is not None
            assert out is None or is_dtype(out, np.uint8)
            out_ = np.flip(seqs_, axis=(length_axis, ohe_axis))
            if out is not None:
                out[:] = out_
                out_ = out
            return out_
        else:
            raise ValueError("Invalid sequence type.")


def _can_build_lut(codons: list[str]) -> bool:
    """True when the standard O(1) LUT path applies: every codon is length-3
    and uses only the four standard nucleotides A, C, G, T. Non-standard
    alphabets (different codon size or extended/IUPAC characters) use the
    generic :func:`gufunc_translate` scan instead."""
    return all(len(c) == 3 and set(c) <= set("ACGT") for c in codons)


def _build_translate_lut(
    codons: list[str], amino_acids: list[str]
) -> NDArray[np.uint8]:
    """Build the 64-entry codon→AA lookup table consumed by ``gufunc_translate_lut``.

    The packed index for each codon is computed by ``_pack_codon_index`` — the
    same hash the runtime uses — so the table and the lookup cannot drift.

    Callers must gate construction with ``_can_build_lut`` (length-3, ACGT-only
    codons); this function does not re-validate. The table is initialised to the
    ``'X'`` (unknown amino acid) byte so that a *partial* standard alphabet — one
    that is ACGT/k=3 but omits some of the 64 codons — resolves missing codons to
    ``'X'`` rather than uninitialised memory. For the complete standard genetic
    code all 64 slots are overwritten.

    Parameters
    ----------
    codons
        List of length-3 DNA strings (uppercase ACGT).
    amino_acids
        List of single-character amino-acid strings, aligned with ``codons``.

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(64,)``; ``lut[idx]`` is the AA byte for the codon at packed
        index ``idx``.
    """
    lut = np.full(64, ord("X"), dtype=np.uint8)
    for codon, aa in zip(codons, amino_acids, strict=True):
        idx = _pack_codon_index(ord(codon[0]), ord(codon[1]), ord(codon[2]))
        lut[idx] = ord(aa)
    return lut


class AminoAlphabet:
    codons: list[str]
    amino_acids: list[str]
    codon_array: NDArray[np.bytes_]
    aa_array: NDArray[np.bytes_]
    codon_to_aa: dict[str, str]
    codon_lut: NDArray[np.uint8] | None
    """Pre-built 64-entry codon→AA LUT for the fast :func:`gufunc_translate_lut`
    path. ``None`` for non-standard alphabets where the codon length isn't
    3 or any codon contains a non-ACGT character — the generic
    :func:`gufunc_translate` path runs in that case."""
    _valid_nuc_bytes: NDArray[np.uint8]
    """Unique nucleotide bytes across all codons (e.g. ``ACGT`` for the standard
    alphabet), used by ``translate(validate=True)`` to check string input."""

    def __init__(self, codons: list[str], amino_acids: list[str]) -> None:
        """Construct an alphabet of amino acids and their mappings to codons.

        Parameters
        ----------
        codons
            List of codons.
        amino_acids
            List of amino acids, in the same order

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        k = len(codons[0])
        if any(len(c) != k for c in codons):
            raise ValueError("Got codons with varying lengths.")
        if any(len(a) != 1 for a in amino_acids):
            raise ValueError("Got amino acid symbols that are multiple characters.")
        if len(codons) != len(amino_acids):
            raise ValueError(
                "Got different number of codons and amino acids for mapping."
            )

        self.codons = codons
        self.amino_acids = amino_acids

        self.codon_array = np.array(codons, "S")[..., None].view("S1")
        self.aa_array = np.array(amino_acids, "S1")

        self.codon_to_aa = dict(zip(codons, amino_acids))

        # Build the 64-entry O(1) lookup table only for the standard ACGT × k=3
        # case; non-standard alphabets use the generic linear-scan path.
        if _can_build_lut(codons):
            self.codon_lut = _build_translate_lut(codons, amino_acids)
        else:
            self.codon_lut = None

        # Unique nucleotide bytes across all codons, for translate(validate=True).
        nuc_chars = sorted({c for codon in codons for c in codon})
        self._valid_nuc_bytes = np.array([ord(c) for c in nuc_chars], dtype=np.uint8)
        # Upper-cased version of _valid_nuc_bytes (via bit-5 flip) for the
        # case-insensitive unknown-codon detection inside ``translate``. For an
        # alphabet whose nucleotides are already uppercase (the standard DNA
        # case), this is identical to ``_valid_nuc_bytes``.
        self._valid_upper_bytes = self._valid_nuc_bytes & np.uint8(0xDF)

    def _check_nuc_bytes(self, buf: NDArray[np.uint8]) -> None:
        """Raise ``ValueError`` if any byte in ``buf`` is outside the alphabet's
        nucleotides. Used by ``translate(validate=True)`` for string input."""
        ok = np.isin(buf, self._valid_nuc_bytes)
        if not bool(ok.all()):
            bad = np.unique(buf[~ok]).tobytes().decode("ascii", "replace")
            allowed = self._valid_nuc_bytes.tobytes().decode("ascii")
            raise ValueError(
                f"translate(validate=True): input contains characters outside "
                f"the alphabet {{{allowed}}}: found {bad!r}."
            )

    def _check_ohe_rows(self, data: NDArray[np.uint8], n_nuc: int) -> None:
        """Raise ``ValueError`` unless every row of OHE ``data`` (alphabet axis
        is axis 1 of the packed flat buffer) is exactly one-hot over ``n_nuc``
        nucleotides. Used by ``translate(validate=True)`` for OHE Ragged input.

        Required because decoding maps an all-zero row to the unknown sentinel
        but a *multi-hot* row silently resolves to a real nucleotide — so a
        decode-then-membership check would not catch it.
        """
        if data.ndim != 2:
            raise ValueError(
                f"translate(validate=True): OHE data must be 2-D (total, n_nuc), "
                f"got shape {data.shape!r}."
            )
        if data.shape[1] != n_nuc:
            raise ValueError(
                f"translate(validate=True): OHE width {data.shape[1]} does not "
                f"match nucleotide alphabet size {n_nuc}."
            )
        sums = data.sum(axis=1, dtype=np.int64)
        if not bool((sums == 1).all()):
            raise ValueError(
                "translate(validate=True): every OHE row must be one-hot "
                "(exactly one 1 per nucleotide position)."
            )

    @overload
    def translate(
        self,
        seqs: StrSeqType,
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        on_unknown: OnUnknown = "error",
        unknown_marker: str = "X",
    ) -> NDArray[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: Ragged[np.bytes_],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        on_unknown: OnUnknown = "error",
        unknown_marker: str = "X",
    ) -> Ragged[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: Ragged[np.uint8],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet,
        truncate_stop: bool = False,
        validate: bool = False,
        on_unknown: OnUnknown = "error",
        unknown_marker: str = "X",
    ) -> Ragged[np.uint8]: ...
    def translate(
        self,
        seqs: StrSeqType | Ragged[np.bytes_] | Ragged[np.uint8],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        on_unknown: OnUnknown = "error",
        unknown_marker: str = "X",
    ) -> NDArray[np.bytes_] | Ragged[np.bytes_] | Ragged[np.uint8]:
        """Translate nucleotide sequences to amino acids.

        Parameters
        ----------
        seqs
            Nucleotide sequences. Ragged inputs must have all lengths divisible by
            the codon size. For OHE Ragged (uint8), nuc_alphabet is required.
        length_axis
            Only used for non-Ragged array input.
        nuc_alphabet
            Required when seqs is a Ragged OHE (uint8) array, to decode OHE → bytes.
        truncate_stop
            When True, each output sequence is truncated at the first stop codon
            (inclusive). Only valid for Ragged input. Default False.
        validate
            When True, raise ValueError if any input nucleotide is outside this
            alphabet (e.g. ``N`` or other non-ACGT bytes; for OHE input, any row
            that is not exactly one-hot). Lowercase ``acgt`` are tolerated by the
            translation kernels (see ``on_unknown`` below) but are still rejected
            by ``validate=True`` because the alphabet itself is uppercase. When
            validation passes, the translation is guaranteed exact. Default False
            (no checking).
        on_unknown
            How to handle codons whose bytes are outside ``{A, C, G, T, a, c, g,
            t}``. One of:

            * ``"pad"`` — emit ``unknown_marker`` once per unknown codon.
            * ``"collapse"`` — emit a single ``unknown_marker`` per consecutive
              run of unknown codons (output is shorter than input). Only valid
              for Ragged input; dense arrays preserve shape so ``collapse`` is
              rejected for them.
            * ``"shorten"`` — drop unknown codons entirely (output is shorter
              than input). Only valid for Ragged input.
            * ``"error"`` — raise ``ValueError`` naming the first unknown codon
              position. Default.

        unknown_marker
            Single ASCII character (e.g. ``"X"``, ``"-"``, ``"?"``) used as the
            sentinel for unknown codons under ``on_unknown="pad"`` or
            ``"collapse"``. Default ``"X"``.

        Returns
        -------
        NDArray[np.bytes_] | Ragged[np.bytes_] | Ragged[np.uint8]
            Translated amino acid sequences in the same container type as the input.
        """
        on_unknown = _validate_on_unknown(on_unknown)
        marker_byte = _validate_unknown_marker(unknown_marker)

        if not isinstance(seqs, Ragged):
            check_axes(seqs, length_axis, False)
            seqs = cast_seqs(seqs)
            if validate:
                self._check_nuc_bytes(seqs.view(np.uint8))
            if on_unknown in ("collapse", "shorten"):
                raise ValueError(
                    f"on_unknown={on_unknown!r} is only supported for Ragged "
                    "input because it changes the output length. For dense "
                    "arrays use 'pad' or 'error'."
                )
            codon_size = self.codon_array.shape[-1]
            if length_axis is None:
                length_axis = -1
            if length_axis < 0:
                length_axis = seqs.ndim + length_axis
            if seqs.shape[length_axis] % codon_size != 0:
                raise ValueError(
                    "Sequence length is not evenly divisible by codon length."
                )
            # sliding_window_view appends the window axis at the end, so the
            # original length_axis position now holds the stride-able codon axis.
            codons = np.lib.stride_tricks.sliding_window_view(
                seqs, window_shape=codon_size, axis=length_axis
            )
            codons = array_slice(codons, length_axis, slice(None, None, codon_size))
            codons_u1 = codons.view(np.uint8)
            if self.codon_lut is not None:
                translated = gufunc_translate_lut(
                    codons_u1,
                    self.codon_lut,
                    marker_byte,
                    axes=[-1, -1, (), ()],  # type: ignore
                )  # (..., n_codons)
            else:
                translated = gufunc_translate(
                    codons_u1,
                    self.codon_array.view(np.uint8),
                    self.aa_array.view(np.uint8),
                    marker_byte,
                    axes=[-1, (-2, -1), (-1), (), ()],  # type: ignore
                )  # (..., n_codons)

            if on_unknown == "error":
                # Compute unknown mask on the input codons directly; cheaper
                # and decoupled from the user-chosen marker char.
                mask = _codon_unknown_mask(codons_u1, self._valid_upper_bytes)
                if mask.any():
                    flat_pos = int(np.argmax(mask.ravel()))
                    raise ValueError(
                        f"translate(on_unknown='error'): unknown codon at flat "
                        f"position {flat_pos} (codon index). Set on_unknown='pad' "
                        f"to allow non-canonical codons."
                    )

            return translated.view("S1")

        # --- Ragged path ---
        # Pack to ListOffsetArray so .data and .offsets are contiguous and valid.
        seqs = seqs.to_packed()

        is_ohe = is_rag_dtype(seqs, np.uint8)

        codon_size = self.codon_array.shape[-1]
        lengths = seqs.lengths.ravel()
        offsets = seqs.offsets  # 1D (n+1,) after to_packed

        if (lengths % codon_size != 0).any():
            raise ValueError(
                "All Ragged sequence lengths must be divisible by codon length."
            )

        n = len(lengths)

        # Decode OHE → bytes if needed. seqs.data is (total, n_nuc) for OHE or (total,) for bytes.
        if is_ohe:
            if nuc_alphabet is None:
                raise ValueError("nuc_alphabet is required for OHE Ragged input.")
            if validate:
                self._check_ohe_rows(seqs.data, len(nuc_alphabet.array))
            nuc_bytes_flat: NDArray[np.bytes_] = nuc_alphabet.decode_ohe(  # type: ignore[union-attr]
                seqs.data, ohe_axis=-1
            )
        else:
            nuc_bytes_flat = seqs.data
            if validate:
                self._check_nuc_bytes(nuc_bytes_flat.view(np.uint8))

        # Translate the entire flat buffer in one vectorized call. This is valid because
        # translation is invariant to splitting/concatenation when all lengths % codon_size == 0.
        # nuc_bytes_flat shape: (total, *trailing) — translation broadcasts over trailing axes.
        total = nuc_bytes_flat.shape[0]
        trailing = nuc_bytes_flat.shape[1:]
        if total > 0:
            codons = np.lib.stride_tricks.sliding_window_view(
                nuc_bytes_flat, codon_size, axis=0
            )
            # sliding_window_view appends window axis at end → slice axis 0 by codon_size
            codons = codons[::codon_size]
            # codons shape: (num_codons, *trailing, codon_size); codon axis is last
            codons_u1 = codons.view(np.uint8)
            translated_flat: NDArray[np.bytes_]
            if self.codon_lut is not None:
                translated_flat = gufunc_translate_lut(
                    codons_u1,
                    self.codon_lut,
                    marker_byte,
                    axes=[-1, -1, (), ()],  # type: ignore
                ).view("S1")  # (num_codons, *trailing)
            else:
                translated_flat = gufunc_translate(
                    codons_u1,
                    self.codon_array.view(np.uint8),
                    self.aa_array.view(np.uint8),
                    marker_byte,
                    axes=[-1, (-2, -1), (-1), (), ()],  # type: ignore
                ).view("S1")  # (num_codons, *trailing)
        else:
            codons_u1 = np.empty((0, *trailing, codon_size), dtype=np.uint8)
            translated_flat = np.empty((0, *trailing), dtype="S1")

        new_offsets = offsets // codon_size  # (n+1,) position-based in translated_flat

        # Apply on_unknown policy. For Ragged, we run per-sequence so that
        # "collapse"/"shorten" can shrink each sequence independently while
        # keeping the offsets array consistent. For OHE Ragged input the
        # trailing axis is consumed by decode_ohe so the codons buffer is 1-D
        # along the codon axis; collapse/shorten therefore work uniformly.
        if on_unknown != "pad":
            if total > 0:
                unknown_mask_flat = _codon_unknown_mask(
                    codons_u1, self._valid_upper_bytes
                )
                # unknown_mask_flat has the same leading shape as translated_flat
                # (i.e. (num_codons, *trailing) collapsed because the codon axis
                # is gone — and for non-OHE trailing is empty).
            else:
                unknown_mask_flat = np.zeros(0, dtype=np.bool_)

            if on_unknown == "error":
                if bool(unknown_mask_flat.any()):
                    pos = int(np.argmax(unknown_mask_flat))
                    raise ValueError(
                        f"translate(on_unknown='error'): unknown codon at flat "
                        f"position {pos} (codon index). Set on_unknown='pad', "
                        f"'collapse', or 'shorten' to allow non-canonical codons."
                    )
            else:
                # collapse / shorten: per-sequence shrink. Build a new flat
                # buffer + new offsets, then continue with truncate_stop /
                # OHE re-encoding using those.
                translated_flat, new_offsets = _shrink_ragged_unknowns(
                    translated_flat, new_offsets, unknown_mask_flat, on_unknown
                )

        if truncate_stop:
            starts = new_offsets[:-1].astype(np.int64)
            full_ends = new_offsets[1:].astype(np.int64)
            ends = _nb_find_stop_ends(
                translated_flat.view(np.uint8), starts, full_ends, np.uint8(ord("*"))
            )
            out_offsets = np.stack(
                [
                    starts,
                    ends,
                ]
            )  # (2, n) — ListArray (non-contiguous view)
        else:
            out_offsets = new_offsets  # 1D — ListOffsetArray

        if is_ohe:
            # OHE input: trailing was the OHE alphabet axis (consumed by decode_ohe),
            # so translated_flat has no trailing — re-encode and reshape.
            n_aa = len(self.aa_array)
            ohe_flat = self.ohe(translated_flat).reshape(-1, n_aa)
            return Ragged.from_offsets(ohe_flat, (n, None, n_aa), out_offsets)
        else:
            return Ragged.from_offsets(
                translated_flat, (n, None, *trailing), out_offsets
            )

    def ohe(self, seqs: StrSeqType) -> NDArray[np.uint8]:
        """One hot encode an amino acid sequence."""
        from .._encoders import ohe as _ohe

        return _ohe(seqs, self)

    def decode_ohe(
        self,
        seqs: NDArray[np.uint8],
        ohe_axis: int,
        unknown_char: str = "X",
    ) -> NDArray[np.bytes_]:
        """Convert an OHE array to an S1 byte array."""
        from .._encoders import decode_ohe as _decode_ohe

        return _decode_ohe(seqs, ohe_axis, self, unknown_char)


DNA = NucleotideAlphabet("ACGT", "TGCA")


# Monkey patch DNA instance with a faster complement function using
# a static, const lookup table. The base method is slower because it uses a
# dynamic lookup table.
def complement_bytes(
    self: NucleotideAlphabet,
    byte_arr: NDArray[np.bytes_],
    out: NDArray[np.bytes_] | None = None,
) -> NDArray[np.bytes_]:
    if out is None:
        _out = out
    else:
        _out = out.view(np.uint8)
    _out = ufunc_comp_dna(byte_arr.view(np.uint8), _out)  # type: ignore
    return _out.view("S1")


DNA._complement_bytes = MethodType(complement_bytes, DNA)
