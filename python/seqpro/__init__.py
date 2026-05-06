from importlib import metadata

from . import alphabets, bed, gtf, rag, transforms
from ._analyzers import gc_content, length, nucleotide_content
from ._encoders import decode_ohe, decode_tokens, ohe, pad_seqs, tokenize
from ._modifiers import bin_coverage, jitter, k_shuffle, random_seqs, reverse_complement
from ._utils import cast_seqs
from .alphabets import AA, DNA, RNA, AminoAlphabet, NucleotideAlphabet

__version__ = metadata.version("seqpro")

__all__ = [
    "AA",
    "DNA",
    "RNA",
    "AminoAlphabet",
    "NucleotideAlphabet",
    "alphabets",
    "bed",
    "bin_coverage",
    "cast_seqs",
    "decode_ohe",
    "decode_tokens",
    "gc_content",
    "gtf",
    "jitter",
    "k_shuffle",
    "length",
    "nucleotide_content",
    "ohe",
    "pad_seqs",
    "rag",
    "random_seqs",
    "reverse_complement",
    "tokenize",
    "transforms",
]
