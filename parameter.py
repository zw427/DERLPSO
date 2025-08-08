from typing import List

import torch
from torch.distributions import Distribution


class Parameter:
    def __init__(self, distributions: List[Distribution]):
        """
        :param distributions: List of torch distributions to sample from.

        Usage: 

        from torch.distributions import Normal, Uniform

        a = Normal(0.0, 1.0)
        b = Normal(0.0, 1.0)
        # this also works: 
        # ab = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        c = Uniform(0, 10)
        distributions = [a, b, c]

        param_sampler = Parameter(distributions)
        samples = param_sampler.sample(100)
        samples.shape
        # torch.Size([100, 3]) 
        """
        self.distributions = distributions

    def sample(self, n: int = 1) -> torch.Tensor:
        """
        Sample n values from each distribution and concatenate results column-wise.
        """
        samples = [dist.sample((n, 1)).squeeze(1) for dist in self.distributions]
        # Ensure each sample is 2D for concatenation
        samples = [s.unsqueeze(1) if s.ndim == 1 else s for s in samples]
        import pdb; pdb.set_trace()
        return torch.cat(samples, dim=1)
