from typing import Dict, List, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

from ._utils import SeqType, StrSeqType

ALPHABETS: Dict[str, Union[NucleotideAlphabet, AminoAlphabet]]

class NucleotideAlphabet:
    alphabet: str
    complement: str
    array: NDArray[np.bytes_]
    complement_map: Dict[str, str]
    complement_map_bytes: Dict[bytes, bytes]
    str_comp_table: Dict[int, str]
    bytes_comp_table: bytes

    def __init__(self, alphabet: str, complement: str) -> None: ...
    def __len__(self): ...
    def bytes_to_ohe(self, arr: NDArray[np.bytes_]) -> NDArray[np.uint8]: ...
    def ohe_to_bytes(
        self, ohe_arr: NDArray[np.uint8], ohe_axis=-1
    ) -> NDArray[np.bytes_]: ...
    def complement_bytes(self, byte_arr: NDArray[np.bytes_]) -> NDArray[np.bytes_]: ...
    def rev_comp_byte(
        self, byte_arr: NDArray[np.bytes_], length_axis: int
    ) -> NDArray[np.bytes_]: ...
    def rev_comp_string(self, string: str) -> str: ...
    def rev_comp_bstring(self, bstring: bytes) -> bytes: ...
    @overload
    def reverse_complement(
        self,
        seqs: Union[str, List[str], NDArray[np.str_]],
    ) -> NDArray[np.bytes_]: ...
    @overload
    def reverse_complement(
        self,
        seqs: NDArray[np.bytes_],
        length_axis: int,
    ) -> NDArray[np.bytes_]: ...
    @overload
    def reverse_complement(
        self,
        seqs: NDArray[np.uint8],
        length_axis: int,
        ohe_axis: int,
    ) -> NDArray[np.uint8]: ...
    @overload
    def reverse_complement(
        self,
        seqs: SeqType,
        length_axis: Optional[int] = None,
        ohe_axis: Optional[int] = None,
    ) -> NDArray[Union[np.bytes_, np.uint8]]: ...

class AminoAlphabet:
    def __init__(self, codons: List[str], amino_acids: List[str]) -> None: ...
    def translate(
        self, seqs: StrSeqType, length_axis: Optional[int] = None
    ) -> NDArray[np.bytes_]: ...
