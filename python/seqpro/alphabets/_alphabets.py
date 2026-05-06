from __future__ import annotations

from types import MethodType
from typing import cast, overload

import awkward as ak
import numpy as np
from numpy.typing import NDArray
from typing_extensions import assert_never

from .._numba import (
    _nb_find_stop_ends,
    gufunc_complement_bytes,
    gufunc_ohe,
    gufunc_ohe_char_idx,
    gufunc_translate,
    ufunc_comp_dna,
)
from .._utils import SeqType, StrSeqType, cast_seqs, check_axes, is_dtype
from ..rag import Ragged


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

        for maybe_complement, complement in zip(alphabet[::-1], complement):
            if maybe_complement != complement:
                raise ValueError("Reverse of alphabet does not yield the complement.")

    def ohe(self, seqs: StrSeqType) -> NDArray[np.uint8]:
        """One hot encode a nucleotide sequence.

        Parameters
        ----------
        seqs

        Returns
        -------
        result
            Ohe hot encoded nucleotide sequences. The last axis is the one hot encoding
            and the second to last axis is the length of the sequence.
        """
        _seqs = cast_seqs(seqs)
        return gufunc_ohe(_seqs.view(np.uint8), self.array.view(np.uint8))

    def decode_ohe(
        self,
        seqs: NDArray[np.uint8],
        ohe_axis: int,
        unknown_char: str = "N",
    ) -> NDArray[np.bytes_]:
        """Convert an OHE array to an S1 byte array.

        Parameters
        ----------
        seqs
        ohe_axis
        unknown_char
            Single character to use for unknown values, by default "N"

        Returns
        -------
        result
        """
        idx = gufunc_ohe_char_idx(seqs, axis=ohe_axis)  # type: ignore

        if ohe_axis < 0:
            ohe_axis_idx = seqs.ndim + ohe_axis
        else:
            ohe_axis_idx = ohe_axis

        shape = *seqs.shape[:ohe_axis_idx], *seqs.shape[ohe_axis_idx + 1 :]

        _alphabet = np.concatenate([self.array, [unknown_char.encode("ascii")]])

        return _alphabet[idx].reshape(shape)

    def complement_bytes(
        self, byte_arr: NDArray[np.bytes_], out: NDArray[np.bytes_] | None = None
    ) -> NDArray[np.bytes_]:
        """Get reverse complement of byte (S1) array.

        Parameters
        ----------
        byte_arr
        """
        if out is None:
            _out = out
        else:
            _out = out.view(np.uint8)
        _out = gufunc_complement_bytes(
            byte_arr.view(np.uint8), self.bytes_comp_array.view(np.uint8), _out
        )
        return _out.view("S1")

    def rev_comp_byte(
        self,
        byte_arr: NDArray[np.bytes_],
        length_axis: int,
        out: NDArray[np.bytes_] | None = None,
    ) -> NDArray[np.bytes_]:
        """Get reverse complement of byte (S1) array.

        Parameters
        ----------
        byte_arr
        """
        out = self.complement_bytes(byte_arr, out)
        return np.flip(out, length_axis)

    def rev_comp_string(self, string: str):
        comp = string.translate(self.str_comp_table)
        return comp[::-1]

    def rev_comp_bstring(self, bstring: bytes):
        comp = bstring.translate(self.bytes_comp_table)
        return comp[::-1]

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
        out: NDArray[np.bytes_] | None = None,
    ) -> NDArray[np.uint8]: ...
    @overload
    def reverse_complement(
        self,
        seqs: SeqType,
        length_axis: int | None = None,
        ohe_axis: int | None = None,
        out: NDArray[np.bytes_] | None = None,
    ) -> NDArray[np.bytes_ | np.uint8]: ...
    def reverse_complement(
        self,
        seqs: SeqType,
        length_axis: int | None = None,
        ohe_axis: int | None = None,
        out: NDArray[np.bytes_] | None = None,
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
        result
            Array of bytes (S1) or uint8 for string or OHE input, respectively.
        """
        check_axes(seqs, length_axis, ohe_axis)

        seqs = cast_seqs(seqs)

        if is_dtype(seqs, np.bytes_):
            if length_axis is None:
                length_axis = -1
            return self.rev_comp_byte(seqs, length_axis, out)
        elif is_dtype(seqs, np.uint8):  # OHE
            assert length_axis is not None
            assert ohe_axis is not None
            _out = np.flip(seqs, axis=(length_axis, ohe_axis))
            if out is not None:
                out[:] = _out
                _out = out
            return _out
        else:
            assert_never(seqs)  # type: ignore


class AminoAlphabet:
    codons: list[str]
    amino_acids: list[str]
    codon_array: NDArray[np.bytes_]
    aa_array: NDArray[np.bytes_]
    codon_to_aa: dict[str, str]

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
        if any([len(c) != k for c in codons]):
            raise ValueError("Got codons with varying lengths.")
        if any([len(a) != 1 for a in amino_acids]):
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

    @overload
    def translate(
        self,
        seqs: StrSeqType,
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
    ) -> NDArray[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: Ragged[np.bytes_],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
    ) -> Ragged[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: Ragged[np.uint8],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet,
        truncate_stop: bool = False,
    ) -> Ragged[np.uint8]: ...
    def translate(
        self,
        seqs: StrSeqType | Ragged[np.bytes_] | Ragged[np.uint8],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
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

        Returns
        -------
        result
        """

        if not isinstance(seqs, Ragged):
            check_axes(seqs, length_axis, False)
            seqs = cast_seqs(seqs)
            codon_size = self.codon_array.shape[-1]
            if length_axis is None:
                length_axis = -1
            if seqs.shape[length_axis] % codon_size != 0:
                raise ValueError(
                    "Sequence length is not evenly divisible by codon length."
                )
            if length_axis < 0:
                length_axis = seqs.ndim + length_axis
            codons = np.lib.stride_tricks.sliding_window_view(
                seqs, window_shape=codon_size, axis=-1
            )[..., ::codon_size, :]
            codon_axis = length_axis + 1
            return gufunc_translate(
                codons.view(np.uint8),
                self.codon_array.view(np.uint8),
                self.aa_array.view(np.uint8),
                axes=[codon_axis, (-2, -1), (-1), ()],  # type: ignore
            ).view("S1")

        # --- Ragged path ---
        # Pack to ListOffsetArray so .data and .offsets are contiguous and valid.
        seqs = Ragged(ak.to_packed(seqs))

        is_ohe = np.issubdtype(seqs.dtype, np.uint8)
        if is_ohe and nuc_alphabet is None:
            raise ValueError("nuc_alphabet is required for OHE Ragged input.")

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
            nuc_bytes_flat: NDArray[np.bytes_] = nuc_alphabet.decode_ohe(  # type: ignore[union-attr]
                seqs.data, ohe_axis=-1
            )
        else:
            nuc_bytes_flat = seqs.data

        # Translate the entire flat buffer in one vectorized call. This is valid because
        # translation is invariant to splitting/concatenation when all lengths % codon_size == 0.
        total = nuc_bytes_flat.shape[0]
        if total > 0:
            codons = np.lib.stride_tricks.sliding_window_view(
                nuc_bytes_flat, codon_size, axis=0
            )[::codon_size, :]  # (total // codon_size, codon_size)
            translated_flat: NDArray[np.bytes_] = gufunc_translate(
                codons.view(np.uint8),
                self.codon_array.view(np.uint8),
                self.aa_array.view(np.uint8),
                axes=[1, (-2, -1), (-1), ()],  # type: ignore
            ).view("S1")  # (total // codon_size,)
        else:
            translated_flat = np.empty(0, dtype="S1")

        new_offsets = offsets // codon_size  # (n+1,) position-based in translated_flat

        if truncate_stop:
            starts = new_offsets[:-1].astype(np.int64)
            full_ends = new_offsets[1:].astype(np.int64)
            ends = _nb_find_stop_ends(
                translated_flat.view(np.uint8), starts, full_ends, np.uint8(ord("*"))
            )
            out_offsets = np.stack(
                [starts, ends]
            )  # (2, n) — ListArray (non-contiguous view)
        else:
            out_offsets = new_offsets  # 1D — ListOffsetArray

        if is_ohe:
            n_aa = len(self.aa_array)
            ohe_flat = self.ohe(translated_flat).flatten()
            return Ragged.from_offsets(ohe_flat, (n, None, n_aa), out_offsets)
        else:
            return Ragged.from_offsets(translated_flat, (n, None), out_offsets)

    def ohe(self, seqs: StrSeqType) -> NDArray[np.uint8]:
        """One hot encode an amino acid sequence.

        Parameters
        ----------
        seqs

        Returns
        -------
        result
            Ohe hot encoded amino acid sequences. The last axis is the one hot encoding
            and the second to last axis is the length of the sequence.
        """
        _seqs = cast_seqs(seqs)
        return gufunc_ohe(_seqs.view(np.uint8), self.aa_array.view(np.uint8))

    def decode_ohe(
        self,
        seqs: NDArray[np.uint8],
        ohe_axis: int,
        unknown_char: str = "X",
    ) -> NDArray[np.bytes_]:
        """Convert an OHE array to an S1 byte array.

        Parameters
        ----------
        seqs
        ohe_axis
        unknown_char
            Single character to use for unknown values, by default "X"

        Returns
        -------
        result
        """
        idx = gufunc_ohe_char_idx(seqs, axis=ohe_axis)  # type: ignore

        if ohe_axis < 0:
            ohe_axis_idx = seqs.ndim + ohe_axis
        else:
            ohe_axis_idx = ohe_axis

        shape = *seqs.shape[:ohe_axis_idx], *seqs.shape[ohe_axis_idx + 1 :]

        _alphabet = np.concatenate([self.aa_array, [unknown_char.encode("ascii")]])

        return _alphabet[idx].reshape(shape)


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


DNA.complement_bytes = MethodType(complement_bytes, DNA)
