import importlib.metadata as metadata

from . import alphabets, bed, transforms
from ._analyzers import gc_content, length, nucleotide_content
from ._encoders import decode_ohe, decode_tokens, ohe, pad_seqs, tokenize
from ._modifiers import bin_coverage, jitter, k_shuffle, random_seqs, reverse_complement
from ._utils import cast_seqs
from .alphabets import AA, DNA, RNA, AminoAlphabet, NucleotideAlphabet

__version__ = metadata.version("seqpro")

__all__ = [
    "cast_seqs",
    "bin_coverage",
    "gc_content",
    "length",
    "nucleotide_content",
    "ohe",
    "decode_ohe",
    "pad_seqs",
    "k_shuffle",
    "reverse_complement",
    "random_seqs",
    "NucleotideAlphabet",
    "AminoAlphabet",
    "alphabets",
    "jitter",
    "DNA",
    "RNA",
    "AA",
    "tokenize",
    "transforms",
    "decode_tokens",
    "bed",
]
