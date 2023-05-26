from typing import List, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

from ._alphabets import NucleotideAlphabet
from ._utils import SeqType

@overload
def reverse_complement(
    seqs: Union[str, List[str], NDArray[np.str_]],
    alphabet: NucleotideAlphabet,
) -> NDArray[np.bytes_]: ...
@overload
def reverse_complement(
    seqs: NDArray[np.bytes_],
    alphabet: NucleotideAlphabet,
    length_axis: int,
) -> NDArray[np.bytes_]: ...
@overload
def reverse_complement(
    seqs: NDArray[np.uint8],
    alphabet: NucleotideAlphabet,
    length_axis: int,
    ohe_axis: int,
) -> NDArray[np.uint8]: ...
@overload
def reverse_complement(
    seqs: SeqType,
    alphabet: NucleotideAlphabet,
    length_axis: Optional[int] = None,
    ohe_axis: Optional[int] = None,
) -> NDArray[Union[np.bytes_, np.uint8]]: ...
@overload
def shuffle(
    seqs: Union[str, List[str], NDArray[np.str_]], *, seed: Optional[int] = None
) -> NDArray[np.bytes_]: ...
@overload
def shuffle(
    seqs: NDArray[np.bytes_], length_axis: int, *, seed: Optional[int] = None
) -> NDArray[np.bytes_]: ...
@overload
def shuffle(
    seqs: NDArray[np.bytes_], length_axis: int, *, seed: Optional[int] = None
) -> NDArray[np.uint8]: ...
@overload
def shuffle(
    seqs: SeqType, length_axis: Optional[int] = None, *, seed: Optional[int] = None
) -> NDArray[Union[np.bytes_, np.uint8]]: ...
def k_shuffle(
    seqs: SeqType,
    k: int,
    length_axis: Optional[int] = None,
    seed: Optional[int] = None,
    alphabet: Optional[NucleotideAlphabet] = None,
): ...
def bin_coverage(
    coverage_array: NDArray[np.number],
    bin_width: int,
    length_axis: int,
    normalize=False,
) -> NDArray[np.number]: ...
