import matplotlib.pyplot as plt

from .._analyzers import gc_content_seqs, nucleotide_content_seqs


def plot_gc_content(seqs, title="", ax=None, figsize=(10, 5)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    gc_contents = gc_content_seqs(seqs, ohe=False)
    ax.hist(gc_contents, bins=100)
    ax.set_xlabel("GC content")
    ax.set_ylabel("Frequency")


def plot_nucleotide_content(seqs, title="", ax=None, figsize=(10, 5)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    nuc_contents = nucleotide_content_seqs(seqs, axis=0, ohe=False, normalize=True)
    ax.plot(nuc_contents.T)
    ax.legend(["A", "C", "G", "T"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
