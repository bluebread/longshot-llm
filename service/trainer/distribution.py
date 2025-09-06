import math
from collections import defaultdict
from itertools import combinations
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Gumbel


class GumbelTopKSubset(Distribution):
    """
    A distribution for sampling k-element subsets from n elements using the Gumbel-Top-k trick.
    
    This distribution samples subsets of size k from n elements where each element i has 
    weight exp(phi[i]). The sampling is done using Gumbel noise added to the logits,
    followed by taking the top-k elements.
    
    The distribution supports batched operations and provides exact log probability
    computation using dynamic programming.
    
    Args:
        phi: Logits/weights for each element, shape (..., n)
        k: Number of elements to select in each subset
        statis_freq: Number of samples for entropy estimation
        validate_args: Whether to validate input arguments
    
    Attributes:
        _phi: The input logits
        _k: Subset size
        _alpha: Exponentiated logits (exp(phi))
        _beta: Softmax probabilities (softmax(phi))
        _gumbel: Standard Gumbel distribution for sampling
    """
    arg_constraints = { 
        "_phi": constraints.real_vector,
    }
    has_rsample = False
    
    def __init__(
        self, 
        phi: torch.Tensor, 
        k: int, 
        validate_args = None
    ):
        """
        Initialize the GumbelTopKSubset distribution.
        
        Args:
            phi: A tensor of shape (..., n) representing the logits/weights for n elements.
                Higher values indicate higher probability of selection.
            k: The number of elements to select in each subset. Must be positive and 
                less than or equal to n.
            validate_args: Whether to validate input arguments. If True, checks that
                inputs satisfy constraints.
        
        Raises:
            TypeError: If phi is not a torch.Tensor or k is not an int.
            ValueError: If phi is not at least 1-dimensional, or k is not positive.
        """
        if not isinstance(phi, torch.Tensor) or not isinstance(k, int):
            raise TypeError("`phi` must be torch.Tensor, and `k` must be int")
        if phi.dim() < 1:
            raise ValueError("`phi` must be at least 1-dimensional")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("`k` must be a positive integer (>=1)")
        
        # Initialize parameters
        self._phi = phi.to()
        self._k = k
        self._gumbel = Gumbel(0, 1)
        self._validate_args = validate_args
        
        # Calculate temporary variables
        self._alpha = torch.exp(phi)
        self._beta = torch.softmax(phi, dim=-1)
        
        # Initialize the distribution
        batch_shape = phi.shape[:-1]
        event_shape = torch.Size([k])
        super().__init__(batch_shape, event_shape, validate_args)
        
        
    def sample(self, sample_shape=torch.Size()):
        """
        Sample k-element subsets using the Gumbel-Top-k trick.
        
        The sampling process:
        1. Add Gumbel noise to each logit phi[i]
        2. Select indices of the k largest perturbed values
        3. Return these indices as the sampled subset
        
        Args:
            sample_shape: Shape of samples to draw. The returned tensor will have
                shape sample_shape + batch_shape + event_shape.
        
        Returns:
            Tensor of shape (..., k) containing indices of selected elements.
            Each row represents one k-element subset sampled from the distribution.
        """
        with torch.no_grad():
            phi = self._phi
            g = self._gumbel
            k = self._k
            x = phi + g.sample(sample_shape + phi.shape)
            
            return torch.topk(x, k, dim=-1).indices
        
        
    def entropy(self):
        """
        Compute the exact entropy of the GumbelTopKSubset distribution.
        
        The entropy measures the uncertainty in sampling k-element subsets from n elements.
        It is computed as H = -Σ p(S) log p(S) where the sum is over all possible 
        k-element subsets S.
        
        This method enumerates all C(n,k) possible subsets, computes their log probabilities
        using the log_prob method, and calculates the entropy using the formula:
        H = -Σ exp(log_p) * log_p
        
        Returns:
            Tensor of shape batch_shape containing the entropy value(s).
            Higher values indicate more uncertainty in subset selection.
        
        Note:
            This exact computation has complexity O(C(n,k)) which becomes expensive
            for large n and k. For n=10, k=5, there are 252 subsets to evaluate.
        """
        n = self._phi.size(-1)
        k = self._k
        batch_shape = self._phi.shape[:-1]
        subsets = list(combinations(range(n), k))
        ss = torch.tensor(subsets)
        
        if batch_shape:
            for _ in batch_shape:
                ss = ss.unsqueeze(1)
            ss = ss.expand(-1, *batch_shape, -1)
            
        x = self.log_prob(ss)
        
        return (- x * torch.exp(x)).sum(dim=0)
    

    def _sum_over_complements(self, values: torch.Tensor):
        """
        Compute sum-over-complements using dynamic programming in parallel.
        
        For each subset S of {0, 1, ..., k-1}, computes the sum of values[i]
        for all i in S. This is done for all 2^k subsets simultaneously using
        bit manipulation and matrix multiplication.
        
        Args:
            values: Tensor of shape (..., k) containing values to sum.
        
        Returns:
            Tensor of shape (..., 2^k) where element [..., mask] contains
            the sum of values[..., i] for all i where bit i is set in mask.
            For example, element [..., 5] (binary 101) contains 
            values[..., 0] + values[..., 2].
        """
        k = values.size(-1)
        masks = torch.arange(1 << k).unsqueeze(1)
        masks = masks.bitwise_and(1 << torch.arange(k)).eq(0).float()

        return values @ masks.transpose(0, 1)


    def _align_dim_except_last(self, A: torch.Tensor, B: torch.Tensor):
        """
        Align dimensions of tensor B to match A's batch dimensions.
        
        This helper function ensures B has the same batch dimensions as A
        (all dimensions except the last one). It handles broadcasting by
        adding singleton dimensions and expanding as needed.
        
        Args:
            A: Reference tensor with target batch dimensions.
            B: Tensor to align, must have compatible batch dimensions.
        
        Returns:
            B expanded to have the same batch dimensions as A, with its
            original last dimension preserved.
        
        Raises:
            ValueError: If B has incompatible prefix dimensions that cannot
                be aligned with A.
        """
        a_prefix = A.shape[:-1]
        b_prefix = B.shape[:-1]

        if len(b_prefix) > len(a_prefix):
            raise ValueError(f"B has too many prefix dims {b_prefix}, cannot align with A {a_prefix}")

        # Handle empty batch dimension case
        if len(b_prefix) == 0 and len(a_prefix) > 0:
            # B has no batch dims, need to add them
            B_expanded = B
            for _ in range(len(a_prefix)):
                B_expanded = B_expanded.unsqueeze(0)
            expand_shape = (*a_prefix, B.shape[-1])
            return B_expanded.expand(expand_shape)
        
        if b_prefix != a_prefix[-len(b_prefix):]:
            raise ValueError(f"Prefix dims do not align: A {a_prefix}, B {b_prefix}")

        num_missing = len(a_prefix) - len(b_prefix)
        B_expanded = B
        for _ in range(num_missing):
            B_expanded = B_expanded.unsqueeze(0)

        expand_shape = (*a_prefix, B.shape[-1])
        B_expanded = B_expanded.expand(expand_shape)

        return B_expanded

    def log_prob(self, value: torch.Tensor):
        """
        Compute the log probability of given k-element subsets.
        
        Uses dynamic programming to compute the exact log probability of
        sampling the given subsets. The computation involves:
        1. Gathering the softmax probabilities for selected indices
        2. Computing sum-over-complements for dynamic programming
        3. Building up probabilities using the DP recurrence relation
        4. Computing final log probability with appropriate normalization
        
        Args:
            value: Tensor of shape (..., k) containing indices of k-element
                subsets. Each row should contain k distinct indices from
                {0, 1, ..., n-1} representing a subset.
        
        Returns:
            Tensor containing log probabilities for each subset.
            Shape matches value.shape[:-1].
        
        Raises:
            AssertionError: If the last dimension of value is not equal to k.
        """
        beta = self._beta
        n = self._phi.size(-1)
        k = self._k
        prob_shape = value.shape[:-1]

        assert value.shape[-1] == k, "The last dimension of `value` must be equal to `k`"

        beta_expanded = self._align_dim_except_last(value, beta)
        v = beta_expanded.gather(dim=-1, index=value)
        soc = self._sum_over_complements(v)
        g = 1 / (1 - soc)

        dp = torch.zeros(*prob_shape, 2 ** n)

        for S in range(2 ** k):
            if S.bit_count() <= 1:
                dp[..., S] = 1
                continue

            Q = [S ^ (1 << i) for i in range(n) if (S & (1 << i)) > 0]
            fQ = dp[..., Q]
            gQ = g[..., Q]
            dp[..., S] = (fQ * gQ).sum(dim=-1)

        x = torch.log(dp[..., (1 << k) - 1])
        x = x + torch.log(v).sum(dim=-1)

        return x


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Test configuration
    phi = torch.randn(5)  # Random logits for 5 elements
    k = 2  # Select top-k elements (subset size)
    num_samples = 10000  # Number of samples to generate for testing
    
    # Create the Gumbel Top-K Subset distribution
    dist = GumbelTopKSubset(phi, k)

    # Generate samples and compute their properties
    samples = dist.sample((num_samples,))
    log_probs = dist.log_prob(samples)
    entropy_estimate = dist.entropy()

    # Display basic results
    print("Phi (logits):")
    print(phi)
    print("Samples:")
    print(samples)
    print("Log probabilities:")
    print(log_probs)
    print("Estimated entropy:")
    print(entropy_estimate)
    
    # Count occurrences of each unique subset
    lookup = defaultdict(int)
    for row in samples:
        s = set(row.tolist())
        lookup[frozenset(s)] += 1
    
    # Compare empirical vs theoretical probabilities
    print("\nSubset probability comparison:")
    print("-" * 60)
    for subset, count in lookup.items():
        # Empirical probability from sampling
        est_prob = count / num_samples
        
        # Theoretical probability from distribution
        logp = dist.log_prob(torch.tensor(list(subset)).unsqueeze(0)).item()
        cal_prob = math.exp(logp)
        
        print(f"Subset: {set(subset)}, Estimated Prob.: {est_prob:.6f}, Calculated Prob.: {cal_prob:.6f}")
    