from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve
from .upper_triangular import full_to_triu, triu_to_full


def shrunk_covariance_inplace(emp_cov, shrinkage=0.1):
    """Same as sklearn.covariance.shrunk_covariance but in-place."""
    # See https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    emp_cov *= (1.0 - shrinkage)
    emp_cov.flat[:: n_features + 1] += shrinkage * mu
    return emp_cov


class Optimizer(ABC):
    """
    Abstract class for optimizers that compute energy coefficients to bias a simulation to match a target distribution.
    """
    _uses_cov = False

    def __init__(self, n: int, initial_alpha: float = 0.0):
        self.alpha = np.ones(n) * initial_alpha
        self.alpha0 = None
        self.t = 1

    @abstractmethod
    def update(self, target, simulation, simulation_cov=None):
        """
        Update the energy coefficients based on the difference between the target and the simulation.
        """
        pass

    def initialize(self):
        """
        Method to prepare for a round of reweighting.

        The default implementation saves the current alpha as the initial alpha to faciliate the computation of
        the difference between the current alpha and the initial alpha for reweighting.

        Subclasses may override this method to perform additional initialization steps, such as resetting
        momentum or other parameters.
        """
        self.alpha0 = self.alpha.copy()

    def get_delta_alpha(self) -> np.ndarray:
        """
        Returns the difference between the current alpha and the initial alpha.
        """
        return self.alpha - self.alpha0

    def get_alpha(self) -> np.ndarray:
        return self.alpha.copy()

    def set_alpha(self, alpha: np.ndarray, t: int):
        """
        Set the energy coefficients and the iteration number.
        """
        self.alpha = alpha.copy()
        self.t = t

    def uses_cov(self) -> bool:
        return self._uses_cov

    def save_state(self, npz_fname: str):
        pass

    def load_state(self, npz_fname: str):
        pass


class NewtonOptimizer(Optimizer):
    """
    Compute energy coefficients needed to bias a simulation producing contact
    frequencies f_sim to one producing f_exp.

    The update is given by:
        $$ alpha = \frac{1}{\beta} B^{-1} ( f_{sim} - f_{exp} )^T $$

    WARNING: This function mutates contacts_cov_sim.

    Note: these force coefficients should accumulate additively. That is,
    if the simulation that produced f_sim included coefficients alpha_0
    and this procedure produces coefficients alpha, then the next simulation
    should be done using coefficients alpha_1 = alpha_0 + alpha.
    """
    _uses_cov = True

    def __init__(self, n: int, initial_alpha: float = -1.0, max_update: float = 1.0):
        super().__init__(n, initial_alpha=initial_alpha)
        self.max_update = max_update

    def update(self, target, simulation, simulation_cov=None):
        simulation_cov = shrunk_covariance_inplace(simulation_cov)
        diff = simulation - target
        new_alpha = solve(simulation_cov, diff.T, assume_a='pos', overwrite_a=True)

        lr = min(1.0, self.max_update / np.max(np.abs(new_alpha)))
        update = lr * new_alpha
        self.alpha += update

        print(f'Updated alpha using lr={lr}, largest update is {np.max(np.abs(update))}', flush=True)
        return self.alpha


class AdamOptimizer(Optimizer):
    _uses_cov = False

    def __init__(self, n: int, initial_alpha: float, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 verbose=True):
        super().__init__(n, initial_alpha=initial_alpha)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.alpha)
        self.v = np.zeros_like(self.alpha)
        self.verbose = verbose
        self.t = 1

    def save_state(self, npz_fname):
        with open(npz_fname, 'wb') as f:
            np.savez(f, alpha=self.alpha, t=self.t, m=self.m, v=self.v)

    def load_state(self, npz_fname):
        state = np.load(npz_fname)
        self.alpha = state['alpha']
        self.t = state['t']
        self.m = state['m']
        self.v = state['v']

    def update(self, target, simulation, simulation_cov=None):
        grad = simulation - target

        # Standard Adam update.
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        self.t += 1
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.alpha += update

        if self.verbose:
            print(f'Updated alpha using lr={self.learning_rate}, '
                  f'largest update is {np.nanmax(np.abs(update))}', flush=True)
        return self.alpha


class GradientDescentOptimizer(Optimizer):
    _uses_cov = False

    def __init__(self, n: int, initial_alpha: float = 0.0, learning_rate=1e-3, beta=0.0,
                 warmup_t=0, regularization=0, verbose=True):
        super().__init__(n, initial_alpha=initial_alpha)
        self.learning_rate = learning_rate
        self.beta = beta
        self.contact_map = np.zeros_like(self.alpha)
        self.warmup_t = warmup_t
        self.regularization = regularization
        self.verbose = verbose

    def save_state(self, npz_fname):
        with open(npz_fname, 'wb') as f:
            np.savez(f, alpha=self.alpha, t=self.t, contact_map=self.contact_map)

    def load_state(self, npz_fname):
        state = np.load(npz_fname)
        self.alpha = state['alpha']
        self.t = state['t']
        self.contact_map = state['contact_map']

    def update_contact_map(self, contact_map: np.array):
        """
        Update the exponential moving average of the contact map.
        """
        t = self.t - self.warmup_t
        if t <= 1:
            self.contact_map = contact_map.copy()
        else:
            self.contact_map *= 1 - self.beta**(t - 1)  # Remove bias correction from previous step.
            self.contact_map = contact_map * (1 - self.beta) + self.contact_map * self.beta
            self.contact_map /= 1 - self.beta**t  # Apply bias correction for current step.
        self.t += 1

    def update(self, target, simulation, simulation_cov=None):
        self.update_contact_map(simulation)
        grad = self.contact_map - target - self.alpha * self.regularization
        update = self.learning_rate * grad

        mask = ~np.isnan(target)
        self.alpha[mask] += update[mask]

        if self.verbose:
            print(f'Updated alpha using lr={self.learning_rate}, '
                  f'largest update is {np.nanmax(np.abs(update))}', flush=True)
        return self.alpha


class FactoredGradientDescentOptimizer(Optimizer):
    _uses_cov = False

    def __init__(self, n: int, initial_alpha: float = 0.0, learning_rate=1e-3, beta=0.0,
                 warmup_t=0, regularization=0, verbose=True):
        super().__init__(n, initial_alpha=initial_alpha)
        self.learning_rate = learning_rate
        self.beta = beta
        self.contact_map = np.zeros_like(self.alpha)
        self.warmup_t = warmup_t
        self.regularization = regularization
        self.verbose = verbose

        # alpha = factors * factors^T
        monomers = triu_to_full(np.zeros(n), k=2).shape[0]
        self.factors = np.zeros(monomers) + initial_alpha / 2

    def get_alpha(self):
        return full_to_triu(self.factors[:, None] + self.factors[None, :], k=2)

    def save_state(self, npz_fname):
        with open(npz_fname, 'wb') as f:
            np.savez(f, factors=self.factors, t=self.t, contact_map=self.contact_map)

    def load_state(self, npz_fname):
        state = np.load(npz_fname)
        self.factors = state['factors']
        self.t = state['t']
        self.contact_map = state['contact_map']

    def update_contact_map(self, contact_map: np.array):
        """
        Update the exponential moving average of the contact map.
        """
        t = self.t - self.warmup_t
        if t <= 1:
            self.contact_map = contact_map.copy()
        else:
            self.contact_map *= 1 - self.beta**(t - 1)  # Remove bias correction from previous step.
            self.contact_map = contact_map * (1 - self.beta) + self.contact_map * self.beta
            self.contact_map /= 1 - self.beta**t  # Apply bias correction for current step.
        self.t += 1

    def update(self, target, simulation, simulation_cov=None):
        self.update_contact_map(simulation)
        target_marginal = np.sum(triu_to_full(target, k=2), axis=0)
        simulation_marginal = np.sum(triu_to_full(self.contact_map, k=2), axis=0)
        grad = simulation_marginal - target_marginal #- self.alpha * self.regularization
        update = self.learning_rate * grad

        mask = ~np.isnan(target_marginal)
        self.factors[mask] += update[mask]
        print(update)

        if self.verbose:
            print(f'Updated factors using lr={self.learning_rate}, '
                  f'largest update is {np.nanmax(np.abs(update))}', flush=True)
        return self.alpha


def get_optimizer(name, optimizer_params, n) -> Optimizer:
    if name == 'adam':
        optimizer = AdamOptimizer(n, **optimizer_params)
    elif name == 'factored_gradient_descent':
        optimizer = FactoredGradientDescentOptimizer(n, **optimizer_params)
    elif name == 'gradient_descent':
        optimizer = GradientDescentOptimizer(n, **optimizer_params)
    elif name == 'newton':
        optimizer = NewtonOptimizer(n, **optimizer_params)
    else:
        raise ValueError(f'Invalid optimizer {name}')
    return optimizer
