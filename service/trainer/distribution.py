import math
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Gumbel, Bernoulli


class GumbelTopK(Distribution):
    arg_constraints = { 
        "_phi": constraints.real_vector,
    }
    has_rsample = False
    
    def __init__(
        self, 
        phi: torch.Tensor, 
        k: int, 
        statis_freq: int = 1536, 
        validate_args = None
    ):
        """
        Initialize the GumbelTopK distribution.
        Args:
            phi (torch.Tensor): A tensor of shape (..., n) representing the weights for the distribution.
            k (int): The number of top elements to sample.
            statis_freq (int): Number of samples to use for estimating entropy.
            validate_args (bool): Whether to validate arguments.
        """
        if not isinstance(phi, torch.Tensor) or not isinstance(k, int):
            raise TypeError("`phi` must be torch.Tensor, and `k` must be int")
        if phi.dim() < 1:
            raise ValueError("`phi` must be at least 1-dimensional")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("`k` must be a positive integer (>=1)")
        if not isinstance(statis_freq, int) or statis_freq <= 0:
            raise ValueError("`statis_freq` must be a positive integer (>=1)")
        
        # Initialize parameters
        self._phi = phi.to()
        self._k = k
        self._gumbel = Gumbel(0, 1)
        self._statis_freq = statis_freq
        self._validate_args = validate_args
        
        # Calculate temporary variables
        self.alpha = torch.exp(phi)
        self.beta = torch.softmax(phi, dim=-1)
        
        # Initialize the distribution
        batch_shape = phi.shape[:-1]
        event_shape = torch.Size([k])
        super().__init__(batch_shape, event_shape, validate_args)
        
        
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the GumbelTopK distribution.
        Args:
            sample_shape (torch.Size): The shape of the samples to draw.
        Returns:
            torch.Tensor: A tensor containing the indices of the top-k elements sampled from the GumbelTopK distribution.
        """
        with torch.no_grad():
            phi = self._phi
            g = self._gumbel
            k = self._k
            x = phi + g.sample(sample_shape + phi.shape)
            
            return torch.topk(x, k, dim=-1).indices
        
        
    def entropy(self):
        """
        Estimate the entropy of the GumbelTopK distribution using sampling.
        Returns:
            torch.Tensor: A tensor containing the estimated entropy of the distribution.
        """
        sample_shape = torch.Size([self._statis_freq])
        x = self.log_prob(self.sample(sample_shape))
        
        return -x.mean(dim=0)
    

    def _sum_over_complements(self, values: torch.Tensor):
        """
        Compute sum-over-complements (SOC) dynamic programming in parallel.

        For each subset S of {0, 1, ..., k-1}, computes the sum of values[i]
        where i is in S. This is done for all 2^k subsets simultaneously.
        
        Args:
            values: Tensor of shape (..., k) containing values to sums
            
        Returns:
            Tensor of shape (..., 2^k) where element [..., mask] contains
            the sum of values[..., i] for all i where bit i is set in mask
        """
        k = values.size(-1)
        masks = torch.arange(1 << k).unsqueeze(1)
        masks = masks.bitwise_and(1 << torch.arange(k)).eq(0).int()

        return values @ masks.transpose(0, 1)


    def _align_dim_except_last(self, A: torch.Tensor, B: torch.Tensor):
        """
        Align the dimensions of tensor B to match those of tensor A, except for the last dimension.
        """
        a_prefix = A.shape[:-1]
        b_prefix = B.shape[:-1]

        if len(b_prefix) > len(a_prefix):
            raise ValueError(f"B has too many prefix dims {b_prefix}, cannot align with A {a_prefix}")

        # 對齊從後面開始比
        if b_prefix != a_prefix[-len(b_prefix):]:
            raise ValueError(f"Prefix dims do not align: A {a_prefix}, B {b_prefix}")

        # 補上缺少的 batch 維度
        num_missing = len(a_prefix) - len(b_prefix)
        B_expanded = B
        for _ in range(num_missing):
            B_expanded = B_expanded.unsqueeze(0)

        # expand 到完整 batch size
        expand_shape = (*a_prefix, B.shape[-1])
        B_expanded = B_expanded.expand(expand_shape)

        return B_expanded

    def log_prob(self, value: torch.Tensor):
        """
        Compute the log probability of a given sample from the GumbelTopK distribution.
        Args:
            value (torch.Tensor): A tensor of shape (..., phi.shape[:-1], k) containing the indices of the top-k elements.
        Returns:
            torch.Tensor: A tensor containing the log probabilities of the given samples.
        """
        phi = self._phi
        n = self._phi.size(-1)
        k = self._k
        prob_shape = value.shape[:-1]

        assert value.shape[-1] == k, "The last dimension of `value` must be equal to `k`"

        phi_expanded = self._align_dim_except_last(value, self._phi)
        v = phi_expanded.gather(dim=-1, index=value)
        soc = self._sum_over_complements(v)
        g = 1 / (1 - soc)

        dp = torch.zeros(*prob_shape, 2 ** n)

        for S in range(2 ** n):
            if S.bit_count() <= 1:
                dp[..., S] = 1

            Q = [S ^ (1 << i) for i in range(n) if (S & (1 << i)) > 0]
            fQ = dp[..., Q]
            gQ = g[..., Q]
            dp[..., S] = (fQ * gQ).sum(dim=-1)

        x = torch.log(dp[..., (1 << n) - 1])
        x = x + torch.log(phi_expanded).sum(dim=-1)

        return x