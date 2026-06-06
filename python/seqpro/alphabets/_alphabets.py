from __future__ import annotations

from types import MethodType
from typing import Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from .._numba import (
    _nb_drop_unknown_codons,
    _nb_find_stop_ends,
    _pack_codon_index,
    gufunc_complement_bytes,
    gufunc_translate,
    gufunc_translate_lut,
    ufunc_comp_dna,
)
from .._utils import SeqType, StrSeqType, array_slice, cast_seqs, check_axes, is_dtype
from ..rag import Ragged, is_rag_dtype


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


def _parse_unknown(unknown: str) -> tuple[bool, np.uint8]:
    """Parse the ``unknown`` policy string for ``AminoAlphabet.translate``.

    Returns ``(is_drop, marker_byte)``. ``"drop"`` -> ``(True, 0)``; a single
    ASCII char -> ``(False, ord(char))``; anything else raises ``ValueError``.
    """
    if unknown == "drop":
        return True, np.uint8(0)
    if isinstance(unknown, str) and len(unknown) == 1 and ord(unknown) <= 0x7F:
        return False, np.uint8(ord(unknown))
    raise ValueError(
        f"unknown must be a single ASCII character (pad) or the literal "
        f'"drop"; got {unknown!r}.'
    )


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

        # Upper-cased view (bit-5 flip) for case-insensitive validation and
        # drop-codon detection. Identical to _valid_nuc_bytes when the
        # alphabet's nucleotides are already uppercase (standard DNA case).
        self._valid_upper_bytes = self._valid_nuc_bytes & np.uint8(0xDF)

    def _check_nuc_bytes(self, buf: NDArray[np.uint8]) -> None:
        """Raise ``ValueError`` if any byte in ``buf`` is outside the alphabet's
        nucleotides. Case-insensitive (``& 0xDF``), matching ``translate``'s
        unconditional case-folding. Used by ``translate(validate=True)``."""
        ok = np.isin(buf & np.uint8(0xDF), self._valid_upper_bytes)
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
        unknown: Literal["drop"],
    ) -> Ragged[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: StrSeqType,
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        unknown: str = "X",
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
        unknown: str = "X",
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
        unknown: str = "X",
    ) -> Ragged[np.uint8]: ...
    def translate(
        self,
        seqs: StrSeqType | Ragged[np.bytes_] | Ragged[np.uint8],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        unknown: str = "X",
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
            Required when seqs is a Ragged OHE (uint8) array, to decode OHE -> bytes.
        truncate_stop
            When True, each output sequence is truncated at the first stop codon
            (inclusive). Only valid for Ragged input. Default False.
        validate
            When True, raise ValueError if any input nucleotide is outside this
            alphabet (case-insensitive: lowercase ``acgt`` pass, ``N`` and other
            non-ACGT bytes raise; for OHE input, any non-one-hot row raises).
            When validation passes, the translation is guaranteed exact. This is
            the single fast-fail path — there is no separate ``error`` policy.
            Default False.
        unknown
            Policy for codons containing a byte outside ``{A, C, G, T, a, c, g,
            t}``. Either a single ASCII character (default ``"X"``) emitted once
            per non-canonical codon ("pad"), or the literal ``"drop"`` which
            removes non-canonical codons entirely. Because ``"drop"`` changes
            per-sequence length, it always returns a ``Ragged`` (even for dense
            input). Case-insensitivity is unconditional and independent of this
            parameter.

        Returns
        -------
        NDArray[np.bytes_] | Ragged[np.bytes_] | Ragged[np.uint8]
            Translated amino acids. Dense input returns a dense array unless
            ``unknown="drop"``, which returns a Ragged.
        """
        is_drop, marker_byte = _parse_unknown(unknown)

        if not isinstance(seqs, Ragged):
            check_axes(seqs, length_axis, False)
            seqs = cast_seqs(seqs)
            if validate:
                self._check_nuc_bytes(seqs.view(np.uint8))
            codon_size = self.codon_array.shape[-1]
            if length_axis is None:
                length_axis = -1
            if length_axis < 0:
                length_axis = seqs.ndim + length_axis
            if seqs.shape[length_axis] % codon_size != 0:
                raise ValueError(
                    "Sequence length is not evenly divisible by codon length."
                )

            if is_drop:
                # Normalize to (n_seq, L): move length axis last, flatten the
                # rest into the batch dim. Each row becomes one Ragged sequence.
                norm = np.moveaxis(seqs, length_axis, -1)
                seq_len = norm.shape[-1]
                norm = np.ascontiguousarray(norm.reshape(-1, seq_len))
                n_seq = norm.shape[0]
                n_codons = seq_len // codon_size
                codons = np.lib.stride_tricks.sliding_window_view(
                    norm, window_shape=codon_size, axis=-1
                )[:, ::codon_size]  # (n_seq, n_codons, codon_size)
                codons_u1 = np.ascontiguousarray(codons.view(np.uint8))
                if self.codon_lut is not None:
                    translated = gufunc_translate_lut(
                        codons_u1,
                        self.codon_lut,
                        marker_byte,
                        axes=[-1, -1, (), ()],  # type: ignore
                    )
                else:
                    translated = gufunc_translate(
                        codons_u1,
                        self.codon_array.view(np.uint8),
                        self.aa_array.view(np.uint8),
                        marker_byte,
                        axes=[-1, (-2, -1), (-1), (), ()],  # type: ignore
                    )
                translated_flat = np.ascontiguousarray(translated.reshape(-1))
                codons_flat = codons_u1.reshape(-1, codon_size)
                offsets = np.arange(n_seq + 1, dtype=np.int64) * n_codons
                # The drop criterion is per-nucleotide-byte validity, which
                # matches the LUT kernel's range-check exactly. For the generic
                # linear-scan kernel this assumes a dense codon table (every
                # all-valid-nucleotide codon has a key) — true for the standard
                # genetic code. A sparse custom alphabet could keep an unkeyed
                # codon still carrying the marker byte.
                out_u1, new_offsets = _nb_drop_unknown_codons(
                    translated_flat, codons_flat, offsets, self._valid_upper_bytes
                )
                return Ragged.from_offsets(
                    out_u1.view("S1"), (n_seq, None), new_offsets
                )

            # pad: shape-preserving dense output
            codons = np.lib.stride_tricks.sliding_window_view(
                seqs, window_shape=codon_size, axis=length_axis
            )
            codons = array_slice(codons, length_axis, slice(None, None, codon_size))
            if self.codon_lut is not None:
                return gufunc_translate_lut(
                    codons.view(np.uint8),
                    self.codon_lut,
                    marker_byte,
                    axes=[-1, -1, (), ()],  # type: ignore
                ).view("S1")
            return gufunc_translate(
                codons.view(np.uint8),
                self.codon_array.view(np.uint8),
                self.aa_array.view(np.uint8),
                marker_byte,
                axes=[-1, (-2, -1), (-1), (), ()],  # type: ignore
            ).view("S1")

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
            codons = codons[::codon_size]
            # codons shape: (num_codons, *trailing, codon_size); codon axis last
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
            codons_u1 = np.empty((0, codon_size), dtype=np.uint8)
            translated_flat = np.empty((0, *trailing), dtype="S1")

        new_offsets = offsets // codon_size  # (n+1,) codon-indexed in translated_flat

        if is_drop:
            # For OHE Ragged the nucleotide-alphabet trailing axis was consumed
            # by decode_ohe, so translated_flat / codons_u1 are 1-D along the
            # codon axis. (Dense drop is handled in the non-Ragged branch.)
            if trailing:
                raise ValueError(
                    "unknown='drop' is not supported for dense-trailing Ragged "
                    "input (e.g. multi-track). Use a single-track Ragged."
                )
            if total > 0:
                out_u1, new_offsets = _nb_drop_unknown_codons(
                    translated_flat.view(np.uint8),
                    np.ascontiguousarray(codons_u1),
                    new_offsets.astype(np.int64),
                    self._valid_upper_bytes,
                )
                translated_flat = out_u1.view("S1")
            else:
                translated_flat = np.empty((0,), dtype="S1")

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
