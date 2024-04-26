import numba as nb
import numpy as np
from numpy.typing import NDArray


class TMM:
    """Trimmed Mean of M-values.

    Parameters
    ----------
    expression_cutoff : float, default -1e10
        Cutoff for expression values.
    log_ratio_trim : float, default 0.3
        Trim fraction for log ratios.
    expression_trim : float, default 0.05
        Trim fraction for absolute expression values.
    apply_weighting : bool, default True
        Whether to weight the scaling factors.
    quantile : float, default 0.75
        Quantile to use for reference sample selection.
    """

    def __init__(
        self,
        expression_cutoff=-1e10,
        log_ratio_trim=0.3,
        expression_trim=0.05,
        apply_weighting=True,
        quantile=0.75,
    ):
        self.expression_cutoff = expression_cutoff
        self.log_ratio_trim = log_ratio_trim
        self.expression_trim = expression_trim
        self.apply_weighting = apply_weighting
        self.quantile = quantile

    @property
    def _is_fitted(self):
        return any(v.endswith("_") for v in vars(self))

    def fit(self, counts: NDArray):
        """
        Fit the TMM normalization model to the given counts.

        Parameters
        ----------
        counts : NDArray
            The data to be normalized. Floating point data is ok.

        Returns
        -------
        self : TMM
            Fitted TMM object.
        """
        # choose reference sample
        # (s)
        normed_quantile = np.quantile(counts, self.quantile, axis=1) / counts.sum(1)
        if np.median(normed_quantile) < 1e-20:
            ref_idx = np.sqrt(counts).sum(1).argmax()
        else:
            ref_idx = np.abs(normed_quantile - normed_quantile.mean()).argmin()

        # (1 g)
        self.ref_counts_ = counts[[ref_idx]]
        self.ref_library_size_ = self.ref_counts_.sum(keepdims=True)

        return self

    def transform(self, counts: NDArray, library_size: float = 1e6):
        """
        Apply the TMM normalization to the given counts.

        Parameters
        ----------
        counts : NDArray
            The raw count data to be normalized. Floating point data is ok.
        library_size : float, optional
            The library size to normalize to. Default is 1e6 i.e. counts per million (CPM).
        Returns
        -------
        normalized_counts : NDArray
            The normalized count data.
        """
        if not self._is_fitted:
            raise ValueError("Must fit before transforming.")

        scaling_factors, library_sizes = self._scaling_factors(counts)

        return counts / (library_sizes * scaling_factors)[:, None] * library_size

    def _scaling_factors(self, counts: NDArray):
        """
        Compute the scaling factors for TMM normalization.

        Parameters
        ----------
        counts : NDArray
            The raw count data. Floating point data is ok.

        Returns
        -------
        scaling_factors : NDArray
            The computed scaling factors.
        library_sizes : NDArray
            The library sizes.
        """
        if not self._is_fitted:
            raise ValueError("Must fit before computing scaling factors.")

        n_samples = counts.shape[0]
        library_sizes = counts.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio: NDArray[np.float64] = np.log2(
                (counts / library_sizes) / (self.ref_counts_ / self.ref_library_size_)
            )
            abs_expression: NDArray[np.float64] = (
                np.log2(counts / library_sizes)
                + np.log2(self.ref_counts_ / self.ref_library_size_)
            ) / 2
            asymp_variance: NDArray[np.float64] = (
                library_sizes - counts
            ) / library_sizes / counts + (
                self.ref_library_size_ - self.ref_counts_
            ) / self.ref_library_size_ / self.ref_counts_

        finite = (
            np.isfinite(log_ratio)
            & np.isfinite(abs_expression)
            & np.isfinite(abs_expression > self.expression_cutoff)
        )

        scaling_factors = np.empty(n_samples, np.float64)
        within_tol: NDArray[np.bool_] = np.nanmax(np.abs(log_ratio), 1) < 1e-6
        scaling_factors[within_tol] = 1

        subset = finite & ~within_tol[:, None]
        n_subset: int = (~within_tol).sum()

        offsets = np.empty(n_subset + 1, np.uint32)
        offsets[0] = 0
        finite[~within_tol, :].sum(1).cumsum(out=offsets[1:])

        log_ratio = log_ratio[subset]
        abs_expression = abs_expression[subset]
        asymp_variance = asymp_variance[subset]

        scaling_factors[~within_tol] = _tmm_helper(
            log_ratio,
            abs_expression,
            asymp_variance,
            offsets,
            self.log_ratio_trim,
            self.expression_trim,
            self.apply_weighting,
        )

        return scaling_factors, library_sizes.squeeze()


@nb.njit(parallel=True, nogil=True, cache=True)
def _tmm_helper(
    log_ratio: NDArray[np.float64],
    abs_expr: NDArray[np.float64],
    var: NDArray[np.float64],
    offsets: NDArray[np.uint32],
    logratioTrim: float,
    sumTrim: float,
    doWeighting: bool,
):
    n_samples = len(offsets) - 1
    scaling_factors = np.empty(n_samples, np.float64)
    for i in nb.prange(n_samples):
        log_r = log_ratio[offsets[i] : offsets[i + 1]]
        abs_e = abs_expr[offsets[i] : offsets[i + 1]]
        v = var[offsets[i] : offsets[i + 1]]
        n = len(log_r)
        lower_log_r = np.floor(n * logratioTrim) + 1
        upper_log_r = n + 1 - lower_log_r
        lower_abs_e = np.floor(n * sumTrim) + 1
        upper_abs_e = n + 1 - lower_abs_e

        logr_ranks = log_r.argsort().argsort()
        absE_ranks = abs_e.argsort().argsort()

        in_range: NDArray[np.bool_] = (
            (logr_ranks >= lower_log_r)
            & (logr_ranks < upper_log_r)
            & (absE_ranks >= lower_abs_e)
            & (absE_ranks < upper_abs_e)
        )

        if doWeighting:
            factor = np.nansum(log_r[in_range] / v[in_range]) / np.nansum(
                1 / v[in_range]
            )
        else:
            factor = np.nanmean(log_r[in_range])

        if np.isnan(factor):
            factor = 0

        scaling_factors[i] = 2**factor
    return scaling_factors
