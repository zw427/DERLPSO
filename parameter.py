from typing import List

import numpy as np

class Parameter:
    def __init__(self, distributions: List):
        """
        :param distributions: List of SciPy continuous distributions to sample from.

        Usage:

        from scipy.stats import Normal, Uniform

        a = Normal(mu = 0.0, sigma = 1.0)
        b = Normal(mu = 0.0, sigma = 1.0)
        # this also works: 
        # ab = Normal(mu = [0.0, 0.0], sigma = [1.0, 1.0])
        c = Uniform(a = 0, b = 10)
        distributions = [a, b, c]

        param_sampler = Parameter(distributions)
        samples = param_sampler.sample(100)
        samples.shape
        # (100, 3)
        """
        self.distributions = distributions

    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample n values from each distribution and concatenate results column-wise.
        """
        samples = [dist.sample((n, 1)).squeeze(1) for dist in self.distributions]
        # Ensure each sample is 2D for concatenation
        samples = [np.expand_dims(s, 1) if s.ndim == 1 else s for s in samples]
        return np.concat(samples, axis=1)
