import matplotlib.pyplot as plt
import numpy as np
from functools import cached_property


class ConvergenceAnalyzer:
    """
    Analyze the convergence of MD/MCMC chains.

    These methods are taken from the STAN library and BDA3 book.
    """
    def __init__(self, chains, pad=100):
        """
        Initialize the ChainConvergenceAnalyzer with the chains to analyze.

        It is expected that any burn-in has been removed from the chains and that the chains have been split
        if desired.

        Parameters:
        chains: np.ndarray
            The chains to analyze. The shape should be (n_chains, n_draws).
        pad: int
            Only compute autocorrelations up to n_draws - pad.
        """
        self.chains = chains
        self.n_chains = self.chains.shape[0]
        self.n_draws = self.chains.shape[1]
        self.pad = pad

    @cached_property
    def W(self) -> float:
        """
        The estimated variance based on the within chain variances.
        """
        return np.mean(np.var(self.chains, ddof=1, axis=1))

    @cached_property
    def B(self) -> float:
        """
        The estimated variance based on the variance between chains.
        """
        return self.n_draws * np.var(np.mean(self.chains, axis=1), ddof=1)

    @cached_property
    def var_hat(self) -> float:
        """
        The estimated variance combining the within and between chain estimates.

        This is our best estimate of the variance.
        """
        return (self.n_draws - 1) / self.n_draws * self.W + self.B / self.n_draws

    @cached_property
    def r_hat(self) -> float:
        """
        The potential scale reduction.

        Values below 1.1 typically indicate adequate mixing.
        """
        return np.sqrt(self.var_hat / self.W)

    @cached_property
    def chain_autocov(self) -> np.ndarray:
        """
        The autocovariance between the samples in each chain.

        Remember that rho(x, y) = cov(x, y) / var(x)var(y).

        This function computes the autocovariances, not the autocorrelations.
        """
        T = self.n_draws - self.pad
        _chains = self.chains - np.mean(self.chains, axis=1, keepdims=True)
        return np.array([np.mean(_chains * _chains, axis=1)]
                        + [np.mean(_chains[:, t:] * _chains[:, :-t], axis=1) for t in range(1, T)]).T

    @cached_property
    def chain_autocorr(self) -> np.ndarray:
        """
        The autocorrellations for each chain for all lags up to self.n_draws - self.pad.
        """
        return self.chain_autocov / np.var(self.chains, axis=1, keepdims=True)

    @cached_property
    def autocorr(self) -> np.ndarray:
        """
        The combined autocorrelation time accounting for the between and within chain variances.

        This is our best estimate of the autocorrelation.
        """
        chain_autocov = self.chain_autocov
        return 1 - (self.W - np.mean(chain_autocov, axis=0)) / self.var_hat

    @cached_property
    def autocorr_time(self) -> float:
        """
        Compute the integrated autocorrelation time, up to the first negative value.
        """
        autocorr = self.autocorr

        if np.all(autocorr >= 0):
            first_negative = -1
        else:
            first_negative = np.argmax(autocorr < 0)
        return 1 + 2 * np.sum(autocorr[1:first_negative])

    @cached_property
    def ess(self) -> float:
        """
        The effective sample size.
        """
        return self.n_draws * self.n_chains / self.autocorr_time

    @cached_property
    def sem(self) -> float:
        """
        The standard error of the mean.
        """
        means = np.mean(self.chains, axis=1)
        return np.std(means) / np.sqrt(self.n_chains)

    @cached_property
    def sem_neff(self) -> float:
        """
        The standard error of the mean.
        """
        return np.std(self.chains) / np.sqrt(self.ess)

    def report(self):
        return (f'Potential Scale Reduction: {self.r_hat:.2f}\n'
                f'Total Sample Size: {self.n_draws * self.n_chains}\n'
                f'Effective Sample Size: {self.ess:.2f}\n'
                f'Autocorrelation Time: {self.autocorr_time:.2f}\n'
                f'Total Deviation: {np.sqrt(self.var_hat):.2f}\n'
                f'Within Chain Deviation: {np.sqrt(self.W):.2f}\n'
                f'Between Chain Deviation: {np.sqrt(self.B):.2f}\n'
                f'Standard Error of the Mean: {self.sem:.2f}, {self.sem_neff:.2f}\n'
                )

    def plot(self, fname=None):
        fig, ax = plt.subplots(2, 1, figsize=(10, 4))

        # Trace Plots
        ax[0].plot(self.chains.T, alpha=0.5)
        ax[0].plot(np.mean(self.chains, axis=0), color='black', lw=3)
        ax[0].axhline(np.mean(self.chains), color='black', linestyle='--')
        ax[0].set_ylabel('Trace')

        # Combined Autocorrelation
        ax[1].plot(self.autocorr, color='blue', alpha=0.5)
        ax[1].axvline(self.autocorr_time, color='red', linestyle='--', label=f'Autocorr Time: {self.autocorr_time:.2f}')
        ax[1].set_ylabel('Autocorrelation')
        ax[1].axhline(0, c='k')
        ax[1].set_ylim(-0.2, 1)
        ax[1].legend()

        ax[0].set_xlim(0)
        ax[1].set_xlim(0)
        plt.tight_layout()

        if fname is not None:
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()
