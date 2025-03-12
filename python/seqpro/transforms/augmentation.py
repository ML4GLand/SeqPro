from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .._modifiers import jitter, k_shuffle, reverse_complement
from .._numba import gufunc_tokenize
from ..alphabets import DNA


class Sequential:
    def __init__(self, *transforms: Callable):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Random:
    def __init__(self, p: float, *transforms: Callable, seed=None):
        self.p = p
        self.transforms = transforms
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        if self.rng.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x


class ReverseComplement:
    def __init__(
        self,
        *types: Literal["dna", "track"],
        length_axis: int,
        ohe_axis: Optional[int],
    ):
        """Reverse complement for DNA sequences or tracks.

        Parameters
        ----------
        types : str
            The type of input. "dna" for DNA sequences and "track" for tracks.
        length_axis : int
            The axis that represents the length of the sequence.
        ohe_axis : int, optional
            The axis that represents the one-hot encoding. Use None for input that is not one-hot encoded.
        """
        self.types = types
        self.length_axis = length_axis
        self.ohe_axis = ohe_axis

    def _rc(self, x: NDArray, type: Literal["dna", "track"]):
        if type == "dna":
            return reverse_complement(x, DNA, self.length_axis, self.ohe_axis)
        elif type == "track":
            return np.flip(x, self.length_axis)

    def __call__(self, *x: NDArray):
        out = tuple(self._rc(_x, t) for _x, t in zip(x, self.types))
        if len(out) == 1:
            out = out[0]
        return out


class KShuffle:
    def __init__(self, k: int, length_axis: int, seed=None) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.length_axis = length_axis

    def __call__(self, x: NDArray) -> NDArray:
        return k_shuffle(
            x,
            self.k,
            length_axis=self.length_axis,
            seed=self.rng.integers(np.iinfo(np.uint32).max, dtype=np.uint32),
        )


class Jitter:
    def __init__(
        self,
        max_jitter: int,
        length_axis: int,
        jitter_axes: Union[int, Tuple[int]],
        seed=None,
    ):
        self.max_jitter = max_jitter
        self.length_axis = length_axis
        self.jitter_axes = jitter_axes
        self.rng = np.random.default_rng(seed)

    def __call__(self, *x: NDArray):
        out = jitter(
            *x,
            max_jitter=self.max_jitter,
            length_axis=self.length_axis,
            jitter_axes=self.jitter_axes,
            seed=self.rng,
        )
        if len(out) == 1:
            out = out[0]
        return out


class Tokenize:
    def __init__(self, token_map: Dict[str, int], unknown_token: int = -1):
        self.token_map = token_map
        self.source = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
        self.target = np.array(list(token_map.values()), dtype=np.int32)
        self.unknown_token = np.int32(unknown_token)

    def __call__(self, x: NDArray):
        return gufunc_tokenize(x, self.source, self.target, self.unknown_token)
