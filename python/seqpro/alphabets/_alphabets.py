from __future__ import annotations

from types import MethodType
from typing import Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from .._numba import (
    _pack_codon_index,
    gufunc_complement_bytes,
    ufunc_comp_dna,
)
from .._utils import SeqType, StrSeqType, cast_seqs, check_axes, is_dtype
from ..rag import Ragged, is_rag_dtype
from ..seqpro import (  # type: ignore[missing-import]  # compiled Rust extension
    _translate_bytes,
    _translate_drop,
    _translate_stop_ends,
    _translate_ohe,
    _translate_ohe_drop,
)


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
    def tokenize(
        self,
        seqs: StrSeqType | Ragged[np.bytes_],
        token_map: dict[str, int],
        unknown_token: int,
        out: NDArray[np.int32] | None = None,
    ) -> NDArray[np.int32] | Ragged[np.int32]:
        """Tokenize sequences using the given token map. Delegates to sp.tokenize."""
        from .._encoders import tokenize as _tokenize

        return _tokenize(seqs, token_map, unknown_token, out)  # pyrefly: ignore[no-matching-overload]  # union call resolved at runtime by overload dispatch

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
    def decode_tokens(
        self,
        seqs: NDArray[np.int32] | Ragged[np.int32],
        token_map: dict[str, int],
        unknown_char: str = "N",
    ) -> NDArray[np.bytes_] | Ragged[np.bytes_]:
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

    def _rust_translate_flat(
        self, nuc_u8: NDArray[np.uint8], marker_byte: np.uint8
    ) -> NDArray[np.uint8]:
        """Translate a contiguous 1-D nucleotide u8 buffer to a flat AA u8 buffer.

        Dispatches to the Rust LUT path for the standard ACGT/k=3 alphabet and
        the Rust linear-scan path otherwise. One AA byte per codon.
        """
        codon_size = self.codon_array.shape[-1]
        flat = np.ascontiguousarray(nuc_u8).reshape(-1)
        if self.codon_lut is not None:
            return _translate_bytes(
                flat, codon_size, self.codon_lut, None, None, int(marker_byte)
            )
        keys = np.ascontiguousarray(self.codon_array.view(np.uint8)).reshape(-1)
        values = np.ascontiguousarray(self.aa_array.view(np.uint8)).reshape(-1)
        return _translate_bytes(flat, codon_size, None, keys, values, int(marker_byte))

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
                n_codons = seq_len // codon_size
                n_seq = int(np.prod(norm.shape[:-1])) if norm.ndim > 1 else 1
                if seqs.size == 0:
                    # Empty input: reshape(-1, 0) and sliding_window_view both
                    # choke on zero-length rows. unknown="drop" always returns a
                    # Ragged, so return an empty one directly.
                    return Ragged.from_offsets(
                        np.empty(0, dtype="S1"),
                        (n_seq, None),
                        np.zeros(n_seq + 1, dtype=np.int64),
                    )
                norm = np.ascontiguousarray(norm.reshape(-1, seq_len))
                codons = np.lib.stride_tricks.sliding_window_view(
                    norm, window_shape=codon_size, axis=-1
                )[:, ::codon_size]  # (n_seq, n_codons, codon_size)
                codons_u1 = np.ascontiguousarray(codons.view(np.uint8))
                translated_flat = self._rust_translate_flat(  # pyrefly: ignore[bad-assignment]
                    norm.reshape(-1).view(np.uint8), marker_byte
                )
                codons_flat = codons_u1.reshape(-1, codon_size)
                offsets = np.arange(n_seq + 1, dtype=np.int64) * n_codons
                # The drop criterion is per-nucleotide-byte validity, which
                # matches the LUT kernel's range-check exactly. For the generic
                # linear-scan kernel this assumes a dense codon table (every
                # all-valid-nucleotide codon has a key) — true for the standard
                # genetic code. A sparse custom alphabet could keep an unkeyed
                # codon still carrying the marker byte.
                out_u1, new_offsets = _translate_drop(
                    translated_flat,
                    np.ascontiguousarray(codons_flat),
                    offsets.astype(np.int64),
                    self._valid_upper_bytes,
                )
                return Ragged.from_offsets(
                    out_u1.view("S1"), (n_seq, None), new_offsets
                )

            # pad: shape-preserving dense output. Move length axis last so codons
            # are contiguous per row, translate the flat buffer, reshape back.
            norm = np.moveaxis(seqs, length_axis, -1)
            lead_shape = norm.shape[:-1]
            seq_len = norm.shape[-1]
            n_codons = seq_len // codon_size
            flat = np.ascontiguousarray(norm.reshape(-1)).view(np.uint8)
            aa = self._rust_translate_flat(flat, marker_byte)  # (total_codons,)
            aa = aa.view("S1").reshape(*lead_shape, n_codons)
            return np.moveaxis(aa, -1, length_axis)

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

        if is_ohe:
            if nuc_alphabet is None:
                raise ValueError("nuc_alphabet is required for OHE Ragged input.")
            if validate:
                self._check_ohe_rows(seqs.data, len(nuc_alphabet.array))
            n_aa = len(self.aa_array)
            nuc_bytes = nuc_alphabet.array.view(np.uint8)
            aa_bytes = self.aa_array.view(np.uint8)
            if self.codon_lut is not None:
                lut_arg, keys_arg, values_arg = self.codon_lut, None, None
            else:
                lut_arg = None
                keys_arg = np.ascontiguousarray(
                    self.codon_array.view(np.uint8)
                ).reshape(-1)
                values_arg = np.ascontiguousarray(self.aa_array.view(np.uint8)).reshape(
                    -1
                )
            if is_drop:
                ohe_flat, new_offsets = _translate_ohe_drop(
                    seqs.data,
                    nuc_bytes,
                    codon_size,
                    self._valid_upper_bytes,
                    offsets.astype(np.int64),
                    lut_arg,
                    keys_arg,
                    values_arg,
                    aa_bytes,
                    int(marker_byte),
                )
            else:
                ohe_flat = _translate_ohe(
                    seqs.data,
                    nuc_bytes,
                    codon_size,
                    lut_arg,
                    keys_arg,
                    values_arg,
                    aa_bytes,
                    int(marker_byte),
                )
                new_offsets = offsets // codon_size
            if truncate_stop:
                # Stop '*' is a real AA row; find its one-hot column and the first
                # row set there (applied AFTER drop-compaction, matching the bytes
                # path's drop-then-truncate order).
                stop_col = int(np.where(aa_bytes == ord("*"))[0][0])
                stop_rows = (ohe_flat[:, stop_col] == 1).view(np.uint8)
                starts = new_offsets[:-1].astype(np.int64)
                full_ends = new_offsets[1:].astype(np.int64)
                ends = _translate_stop_ends(stop_rows, starts, full_ends, np.uint8(1))
                out_offsets = np.stack([starts, ends])
            else:
                out_offsets = new_offsets
            return Ragged.from_offsets(ohe_flat, (n, None, n_aa), out_offsets)

        # --- bytes-only path ---
        nuc_bytes_flat = seqs.data
        if validate:
            self._check_nuc_bytes(nuc_bytes_flat.view(np.uint8))

        # Translate the entire flat buffer in one vectorized call. This is valid because
        # translation is invariant to splitting/concatenation when all lengths % codon_size == 0.
        # nuc_bytes_flat shape: (total, *trailing) — translation broadcasts over trailing axes.
        total = nuc_bytes_flat.shape[0]
        trailing = nuc_bytes_flat.shape[1:]
        translated_flat: NDArray[np.bytes_]
        if total > 0 and not trailing:
            translated_flat = self._rust_translate_flat(
                nuc_bytes_flat.view(np.uint8), marker_byte
            ).view("S1")
        elif total > 0 and trailing:
            # Multi-track bytes: translate each trailing column via the flat
            # codon stride. Move codon axis contiguous per column.
            n_codons = total // codon_size
            cols = int(np.prod(trailing))
            buf = np.ascontiguousarray(
                np.moveaxis(nuc_bytes_flat.reshape(total, cols), 0, -1)
            ).reshape(-1)  # (cols*total,)
            aa = self._rust_translate_flat(buf.view(np.uint8), marker_byte)
            aa = aa.view("S1").reshape(cols, n_codons)
            translated_flat = np.ascontiguousarray(np.moveaxis(aa, 0, -1)).reshape(
                n_codons, *trailing
            )
        else:
            translated_flat = np.empty((0, *trailing), dtype="S1")
        codons_u1 = (
            np.ascontiguousarray(nuc_bytes_flat.view(np.uint8)).reshape(
                total // codon_size if total else 0, codon_size
            )
            if not trailing
            else np.empty((0, codon_size), dtype=np.uint8)
        )

        new_offsets = offsets // codon_size  # (n+1,) codon-indexed in translated_flat

        if is_drop:
            if trailing:
                raise ValueError(
                    "unknown='drop' is not supported for dense-trailing Ragged "
                    "input (e.g. multi-track). Use a single-track Ragged."
                )
            if total > 0:
                out_u1, new_offsets = _translate_drop(
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
            ends = _translate_stop_ends(
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

        return Ragged.from_offsets(translated_flat, (n, None, *trailing), out_offsets)

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
