try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "seqpro"
__version__ = importlib_metadata.version(package_name)

from . import alphabets
from ._analyzers import gc_content, length, nucleotide_content
from ._cleaners import remove_N_seqs, remove_only_N_seqs, sanitize_seq, sanitize_seqs
from ._encoders import decode_ohe, ohe, pad_seqs
from ._modifiers import bin_coverage, jitter, k_shuffle, random_seqs, reverse_complement
from ._utils import cast_seqs
from .alphabets import AminoAlphabet, NucleotideAlphabet

__all__ = [
    "cast_seqs",
    "bin_coverage",
    "gc_content",
    "length",
    "nucleotide_content",
    "remove_N_seqs",
    "remove_only_N_seqs",
    "sanitize_seq",
    "sanitize_seqs",
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
]
