from typing import Dict, List, Optional, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

from .._numba import gufunc_ohe, gufunc_ohe_char_idx, gufunc_translate
from .._utils import SeqType, StrSeqType, cast_seqs, check_axes


class NucleotideAlphabet:
    alphabet: str
    """Alphabet excluding ambiguous characters e.g. "N" for DNA."""
    complement: str
    array: NDArray[np.bytes_]
    complement_map: Dict[str, str]
    complement_map_bytes: Dict[bytes, bytes]
    str_comp_table: Dict[int, str]
    bytes_comp_table: bytes

    def __init__(self, alphabet: str, complement: str) -> None:
        """Parse and validate sequence alphabets.

        Nucleic acid alphabets must be complemented by being reversed (without the
        unknown character). For example, `reverse(ACGT) = complement(ACGT) = TGCA`.

        Parameters
        ----------
        alphabet : str
            For example, DNA could be 'ACGT'.
        complement : str
            Complement of the alphabet, to continue the example this would be 'TGCA'.
        """
        self._validate(alphabet, complement)
        self.alphabet = alphabet
        self.complement = complement
        self.array = cast(
            NDArray[np.bytes_], np.frombuffer(self.alphabet.encode("ascii"), "|S1")
        )
        self.complement_map: Dict[str, str] = dict(
            zip(list(self.alphabet), list(self.complement))
        )
        self.complement_map_bytes = {
            k.encode("ascii"): v.encode("ascii") for k, v in self.complement_map.items()
        }
        self.str_comp_table = str.maketrans(self.complement_map)
        self.bytes_comp_table = bytes.maketrans(
            self.alphabet.encode("ascii"), self.complement.encode("ascii")
        )

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
        seqs : str, list[str], ndarray[str, bytes]

        Returns
        -------
        NDArray[np.uint8]
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
        seqs : NDArray[np.uint8]
        ohe_axis : int
        unknown_char : str, optional
            Single character to use for unknown values, by default "N"

        Returns
        -------
        NDArray[np.bytes_]
        """
        idx = gufunc_ohe_char_idx(seqs, axis=ohe_axis)  # type: ignore

        if ohe_axis < 0:
            ohe_axis_idx = seqs.ndim + ohe_axis
        else:
            ohe_axis_idx = ohe_axis

        shape = *seqs.shape[:ohe_axis_idx], *seqs.shape[ohe_axis_idx + 1 :]

        _alphabet = np.concatenate([self.array, [unknown_char.encode("ascii")]])

        return _alphabet[idx].reshape(shape)

    def complement_bytes(self, byte_arr: NDArray[np.bytes_]) -> NDArray[np.bytes_]:
        """Get reverse complement of byte (S1) array.

        Parameters
        ----------
        byte_arr : ndarray[bytes]
        """
        # * a vectorized implementation using np.unique or np.char.translate is NOT
        # * faster even for longer alphabets like IUPAC DNA/RNA. Another optimization to
        # * try would be using vectorized bit manipulations.
        out = byte_arr.copy()
        for nuc, comp in self.complement_map_bytes.items():
            out[byte_arr == nuc] = comp
        return out

    def rev_comp_byte(
        self, byte_arr: NDArray[np.bytes_], length_axis: int
    ) -> NDArray[np.bytes_]:
        """Get reverse complement of byte (S1) array.

        Parameters
        ----------
        byte_arr : ndarray[bytes]
        """
        out = self.complement_bytes(byte_arr)
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
        length_axis: Optional[int] = None,
        ohe_axis: Optional[int] = None,
    ) -> NDArray[np.bytes_]: ...
    @overload
    def reverse_complement(
        self,
        seqs: NDArray[np.uint8],
        length_axis: Optional[int] = None,
        ohe_axis: Optional[int] = None,
    ) -> NDArray[np.uint8]: ...
    def reverse_complement(
        self,
        seqs: SeqType,
        length_axis: Optional[int] = None,
        ohe_axis: Optional[int] = None,
    ) -> NDArray[Union[np.bytes_, np.uint8]]:
        """Reverse complement a sequence.

        Parameters
        ----------
        seqs : str, list[str], ndarray[str, bytes, uint8]
        length_axis : Optional[int], optional
            Length axis, by default None
        ohe_axis : Optional[int], optional
            One hot encoding axis, by default None

        Returns
        -------
        ndarray[bytes, uint8]
            Array of bytes (S1) or uint8 for string or OHE input, respectively.
        """
        check_axes(seqs, length_axis, ohe_axis)

        seqs = cast_seqs(seqs)

        if seqs.dtype == np.uint8:  # OHE
            assert length_axis is not None
            assert ohe_axis is not None
            return np.flip(seqs, axis=(length_axis, ohe_axis))
        else:
            if length_axis is None:
                length_axis = -1
            return self.rev_comp_byte(seqs, length_axis)  # type: ignore


class AminoAlphabet:
    codons: List[str]
    amino_acids: List[str]
    codon_array: NDArray[np.bytes_]
    aa_array: NDArray[np.bytes_]
    codon_to_aa: Dict[str, str]

    def __init__(self, codons: List[str], amino_acids: List[str]) -> None:
        """Construct an alphabet of amino acids and their mappings to codons.

        Parameters
        ----------
        codons : List[str]
            List of codons.
        amino_acids : List[str]
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

    def translate(
        self, seqs: StrSeqType, length_axis: Optional[int] = None
    ) -> NDArray[np.bytes_]:
        """Translate nucleotide sequences to amino acids.

        Parameters
        ----------
        seqs : StrSeqType
            Nucleotide sequences
        length_axis : Optional[int], optional

        Returns
        -------
        NDArray[np.bytes_]
            Amino acid sequences.
        """
        # TODO this doesn't respect start and stop codons, doing so would also require ragged arrays
        check_axes(seqs, length_axis, False)

        seqs = cast_seqs(seqs)

        codon_size = self.codon_array.shape[-1]

        if length_axis is None:
            length_axis = -1

        if seqs.shape[length_axis] % codon_size != 0:
            raise ValueError("Sequence length is not evenly divisible by codon length.")

        if length_axis < 0:
            length_axis = seqs.ndim + length_axis

        # get k-mers (codons)
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

    def ohe(self, seqs: StrSeqType) -> NDArray[np.uint8]:
        """One hot encode an amino acid sequence.

        Parameters
        ----------
        seqs : str, list[str], ndarray[str, bytes]

        Returns
        -------
        NDArray[np.uint8]
            Ohe hot encoded amino acid sequences. The last axis is the one hot encoding
            and the second to last axis is the length of the sequence.
        """
        _seqs = cast_seqs(seqs)
        return gufunc_ohe(_seqs.view(np.uint8), self.aa_array.view(np.uint8))

    def decode_ohe(
        self,
        seqs: NDArray[np.uint8],
        ohe_axis: int,
        unknown_char: str = "N",
    ) -> NDArray[np.bytes_]:
        """Convert an OHE array to an S1 byte array.

        Parameters
        ----------
        seqs : NDArray[np.uint8]
        ohe_axis : int
        unknown_char : str, optional
            Single character to use for unknown values, by default "N"

        Returns
        -------
        NDArray[np.bytes_]
        """
        idx = gufunc_ohe_char_idx(seqs, axis=ohe_axis)  # type: ignore

        if ohe_axis < 0:
            ohe_axis_idx = seqs.ndim + ohe_axis
        else:
            ohe_axis_idx = ohe_axis

        shape = *seqs.shape[:ohe_axis_idx], *seqs.shape[ohe_axis_idx + 1 :]

        _alphabet = np.concatenate([self.aa_array, [unknown_char.encode("ascii")]])

        return _alphabet[idx].reshape(shape)
