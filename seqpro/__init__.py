try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "seqpro"
__version__ = importlib_metadata.version(package_name)

from ._alphabets import ALPHABETS, AminoAlphabet, NucleotideAlphabet
from ._analyzers import gc_content, length, nucleotide_content
from ._cleaners import remove_N_seqs, remove_only_N_seqs, sanitize_seq, sanitize_seqs
from ._encoders import ohe, ohe_to_bytes, pad_seqs
from ._modifiers import bin_coverage, k_shuffle, reverse_complement
from ._utils import cast_seqs, random_seq, random_seqs

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
    "ohe_to_bytes",
    "pad_seqs",
    "k_shuffle",
    "reverse_complement",
    "random_seq",
    "random_seqs",
    "NucleotideAlphabet",
    "AminoAlphabet",
    "ALPHABETS",
]
