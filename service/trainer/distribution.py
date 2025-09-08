import math
from collections import defaultdict
from itertools import combinations
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Gumbel, Bernoulli


class GumbelTopKSubsetWithSign(Distribution):
    """
    Distribution for sampling signed k-element subsets using Gumbel-Top-k with sign modulation.
    
    This distribution combines subset selection with sign assignment, where:
    1. A k-element subset is sampled from n elements using Gumbel-Top-k trick based on phi logits
    2. Each selected element is assigned a sign (±1) based on psi logits via Bernoulli sampling
    
    The output is a tensor of shape (..., 2k) containing [indices, signs] where indices
    indicate selected elements and signs indicate their polarity (0 or 1).
    
    Mathematical formulation:
    - Subset selection: P(S) ∝ Π_{i∈S} exp(phi[i]) using Gumbel-max trick
    - Sign assignment: P(sign[i]=1) = sigmoid(psi[i]) for each selected element i
    - Joint probability: P(S, signs) = P(S) * Π_{i∈S} P(sign[i])
    
    Args:
        phi: Selection logits for each element, shape (..., n)
        psi: Sign logits for each element, shape (..., n), must be non-negative
        k: Number of elements to select (subset size)
        validate_args: Whether to validate input arguments
    
    Attributes:
        _phi: Selection logits for subset sampling
        _psi: Sign logits for Bernoulli sign assignment  
        _k: Number of elements in each subset
        _beta: Softmax probabilities from phi
        _gumbel: Standard Gumbel(0,1) for noise injection
        _device: Device placement for tensors
    """
    arg_constraints = { 
        "_phi": constraints.real_vector,
    }
    has_rsample = False
    
    def __init__(
        self, 
        phi: torch.Tensor, 
        psi: torch.Tensor, 
        k: int, 
        validate_args = None
    ):
        """
        Initialize the GumbelTopKSubsetWithSign distribution.
        
        Args:
            phi: Selection logits of shape (..., n) for n elements.
                Higher values increase selection probability.
            psi: Sign logits of shape (..., n) for sign assignment.
                Must match phi's shape. Used as Bernoulli logits for sign sampling.
            k: Number of elements to select, must satisfy 1 <= k <= n.
            validate_args: Enable input validation if True.
        
        Raises:
            TypeError: If phi/psi are not torch.Tensor or k is not int.
            ValueError: If shapes don't match, dimensions invalid, or k out of range.
        """
        if not isinstance(phi, torch.Tensor) or not isinstance(k, int):
            raise TypeError("`phi` must be torch.Tensor, and `k` must be int")
        if phi.dim() < 1:
            raise ValueError("`phi` must be at least 1-dimensional")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("`k` must be a positive integer (>=1)")
        if phi.shape != psi.shape:
            raise ValueError("`phi` and `psi` must have the same shape")
        
        # Initialize parameters
        self._phi = phi
        self._psi = psi # must be [0, +inf]
        self._k = k
        self._gumbel = Gumbel(0, 1)
        self._validate_args = validate_args
        self._device = phi.device
        
        # Calculate temporary variables
        self._beta = torch.softmax(phi, dim=-1)
        
        # Initialize the distribution
        batch_shape = phi.shape[:-1]
        event_shape = torch.Size([k])
        super().__init__(batch_shape, event_shape, validate_args)
        
        
    def sample(self, sample_shape=torch.Size()):
        """
        Sample signed k-element subsets via Gumbel-Top-k and Bernoulli sampling.
        
        Sampling algorithm:
        1. Add Gumbel(0,1) noise to phi logits
        2. Select top-k indices based on perturbed values
        3. Extract corresponding psi values for selected indices
        4. Sample signs from Bernoulli(sigmoid(psi)) for each selected element
        5. Concatenate indices and signs into output tensor
        
        Args:
            sample_shape: Additional sample dimensions to prepend.
        
        Returns:
            Tensor of shape (..., 2k) containing [indices, signs].
            First k elements are selected indices, last k are binary signs.
        """
        with torch.no_grad():
            phi = self._phi
            g = self._gumbel
            k = self._k
            noise = g.sample(sample_shape + phi.shape).to(self._device)
            x = phi + noise
            idx = torch.topk(x, k, dim=-1).indices
            psi_extended = self._align_dim_except_last(idx, self._psi)
            psi_sub = psi_extended.gather(dim=-1, index=idx)
            bernoulli = Bernoulli(logits=psi_sub)
            sgn = bernoulli.sample().to(self._device)
            
            return torch.cat((idx, sgn), dim=-1).to(torch.int64)
        
        
    def entropy(self):
        """
        Compute exact entropy of the joint distribution over subsets and signs.
        
        Calculates total entropy as H(S,signs) = H(S) + H(signs|S) where:
        - H(S): Entropy of subset selection via enumeration of C(n,k) subsets
        - H(signs|S): Expected entropy of sign assignments given selected subset
        
        The computation:
        1. Enumerates all possible k-subsets
        2. Computes subset selection entropy: -Σ P(S) log P(S)
        3. Computes conditional sign entropy: Σ P(S) * H(signs|S)
        4. Returns total entropy as sum of both components
        
        Returns:
            Tensor of shape batch_shape with total entropy values.
        
        Complexity:
            O(C(n,k)) - becomes expensive for large n,k combinations.
        """
        n = self._phi.size(-1)
        k = self._k
        batch_shape = self._phi.shape[:-1]
        subsets = list(combinations(range(n), k))
        ss = torch.tensor(subsets, device=self._device, dtype=torch.long)
        
        if batch_shape:
            for _ in batch_shape:
                ss = ss.unsqueeze(1)
            ss = ss.expand(-1, *batch_shape, -1)
            
        x = self._log_prob_subset(ss)
        y = torch.exp(x)
        psi_extended = self._align_dim_except_last(ss, self._psi)
        psi_sub = psi_extended.gather(dim=-1, index=ss)
        bernoulli = Bernoulli(logits=psi_sub)
        
        es = (- x * y).sum(dim=0) # entropy of subset selection0
        eb = (y * bernoulli.entropy().sum(dim=-1)).sum(dim=0) # entropy of sign selection
        
        # H(A, B) = H(A) + H(B|A)
        return es + eb
    

    def _sum_over_complements(self, values: torch.Tensor):
        """
        Parallel computation of subset sums via bit manipulation.
        
        Efficiently computes sums for all 2^k possible subsets of k elements
        using vectorized operations. Each subset is represented by a bitmask.
        
        Algorithm:
        - Creates 2^k x k binary matrix where row i represents subset mask i
        - Matrix multiplication computes all subset sums in parallel
        
        Args:
            values: Tensor (..., k) with values to aggregate.
        
        Returns:
            Tensor (..., 2^k) where position [..., mask] contains sum of
            values at indices where mask has bit set. Example: mask=5 (0b101)
            yields values[...,0] + values[...,2].
        """
        k = values.size(-1)
        masks = torch.arange(1 << k, device=values.device).unsqueeze(1)
        masks = masks.bitwise_and(1 << torch.arange(k, device=values.device)).eq(0).float()

        return values @ masks.transpose(0, 1)


    def _align_dim_except_last(self, A: torch.Tensor, B: torch.Tensor):
        """
        Broadcast B to match A's batch dimensions while preserving last dimension.
        
        Handles dimension alignment for gather operations by:
        1. Adding singleton dimensions to B as needed
        2. Expanding B to match A's batch shape
        3. Preserving B's original last dimension
        
        Args:
            A: Target tensor defining batch shape.
            B: Source tensor to align.
        
        Returns:
            B broadcasted to shape (*A.shape[:-1], B.shape[-1]).
        
        Raises:
            ValueError: If B's prefix dims incompatible with A's batch dims.
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

    def _log_prob_subset(self, idx: torch.Tensor):
        """
        Compute exact log probability of k-element subsets via dynamic programming.
        
        Implements the recurrence relation for subset probabilities:
        - f({i}) = 1 for singleton sets
        - f(S) = Σ_{j∈S} f(S\{j}) * g(S\{j}) for |S| > 1
        - g(S) = 1 / (1 - Σ_{i∉S} β[i]) where β = softmax(φ)
        - P(S) = f(S) * Π_{i∈S} β[i]
        
        The DP builds up probabilities for progressively larger subsets
        using bit manipulation to track subset membership.
        
        Args:
            idx: Tensor (..., k) with k distinct indices per subset.
        
        Returns:
            Log probabilities with shape idx.shape[:-1].
        
        Raises:
            AssertionError: If idx's last dimension ≠ k.
        """
        beta = self._beta
        k = self._k
        # n = self._phi.size(-1)
        # prob_shape = idx.shape[:-1]

        assert idx.shape[-1] == k, "The last dimension of `value` must be equal to `k`"

        beta_expanded = self._align_dim_except_last(idx, beta)
        v = beta_expanded.gather(dim=-1, index=idx)
        # soc = self._sum_over_complements(v)
        # g = 1 / (1 - soc)

        # dp = torch.zeros(*prob_shape, 2 ** n, device=self._device)

        # for S in range(2 ** k):
        #     if S.bit_count() <= 1:
        #         dp[..., S] = 1
        #         continue

        #     Q = [S ^ (1 << i) for i in range(n) if (S & (1 << i)) > 0]
        #     fQ = dp[..., Q]
        #     gQ = g[..., Q]
        #     dp[..., S] = (fQ * gQ).sum(dim=-1)

        # x = torch.log(dp[..., (1 << k) - 1])
        # x = x + torch.log(v).sum(dim=-1)

        # Let Y = {i1, i2, ..., im} be a subset of {1, 2, ..., k}
        # The definition of f:
        #   - |Y| > 1: f(Y) = Σ_{j=1}^{m} f(Y\{ij}) * g(Y\{ij})
        #   - |Y| = 1: f(Y) = 1
        # The definition of g:
        #   - Y = ∅: g(Y) = inf
        #   - |Y| > 0: g(Y) = 1 / (1 - Σ beta[ij] for ij not in Y)
        #   - Y = {1, 2,..., k}: g(Y) = 1 / (1 - 0) = 1
        # The probability of selecting subset S = {i1, i2,...,ik} is:
        #   - p(S) = f({1, 2,...,k}) * Π beta[i]
        #   - log p(S) = log f({1, 2,...,k}) + Σ log beta[i]
        # return x
        return torch.log(v).sum(dim=-1)
    
    def log_prob(self, value) -> torch.Tensor:
        """
        Compute log probability of signed subset samples.
        
        Decomposes the joint probability as:
        log P(indices, signs) = log P(indices) + log P(signs|indices)
        
        Args:
            value: Tensor (..., 2k) containing [indices, signs].
                First k elements are subset indices, last k are binary signs.
        
        Returns:
            Log probability tensor with shape value.shape[:-1].
        
        Raises:
            AssertionError: If value's last dimension ≠ 2k.
        """
        assert value.shape[-1] == 2 * self._k, "The last dimension of `value` must be equal to 2*k"
        k = self._k
        idx, sgn = torch.split(value, [k, k], dim=-1)
        psi_extended = self._align_dim_except_last(idx, self._psi)
        psi_sub = psi_extended.gather(dim=-1, index=idx)
        bernoulli = Bernoulli(logits=psi_sub)
        logp_idx = self._log_prob_subset(idx)
        logp_sgn = bernoulli.log_prob(sgn.float()).sum(dim=-1)
        
        return logp_idx + logp_sgn


class GateTokenDistribution(Distribution):
    """
    Distribution for gate token generation combining type and literal sampling.
    
    Generates tokens consisting of:
    1. Token type: Binary indicator sampled from Bernoulli(sigmoid(zeta))
    2. Literals: Signed k-subset sampled from GumbelTopKSubsetWithSign
    
    The output format is [type, indices, signs] with total dimension 2k+1.
    """
    arg_constraints = { 
        "_phi": constraints.real_vector,
    }
    has_rsample = False
    
    def __init__(
        self, 
        param: torch.Tensor, 
        k: int, 
        validate_args = None
    ):
        """
        Initialize the GateTokenDistribution.
        
        Args:
            param: Parameter tensor (..., 2n+1) containing [zeta, phi, psi] where:
                - zeta (1): Bernoulli logit for token type
                - phi (n): Selection logits for subset sampling
                - psi (n): Sign logits for sign assignment
            k: Subset size for literal sampling, must satisfy 1 <= k <= n.
            validate_args: Enable input validation if True.
        
        Raises:
            AssertionError: If param dimensions invalid or k out of range.
        """
        n = param.size(-1) // 2
        
        assert param.size(-1) == 2 * n + 1, "The last dimension of `param` must be equal to 2n+1"
        assert 1 <= k <= n, "`k` must be a positive integer (>=1) and less than or equal to n"
        
        zeta, phi, psi = torch.split(param, [1, n, n], dim=-1)
        zeta = zeta.squeeze(-1)
        
        self._n = n
        self._k = k
        self.ttype_dist = Bernoulli(logits=zeta)
        self.literals_dist = GumbelTopKSubsetWithSign(phi, psi, k, validate_args)
        
        
    def sample(self, sample_shape=torch.Size()):
        """
        Sample gate tokens with type and signed literals.
        
        Returns:
            Tensor (..., 2k+1) containing [type, indices, signs].
        """
        t = self.ttype_dist.sample(sample_shape).to(self._device)
        l = self.literals_dist.sample(sample_shape).to(self._device)
        
        return torch.cat((t.unsqueeze(-1), l), dim=-1)  
        
        
    def entropy(self):
        """
        Compute total entropy as sum of type and literal entropies.
        
        Returns:
            Total entropy H(type) + H(literals).
        """
        return self.ttype_dist.entropy() + self.literals_dist.entropy()
    
    
    def log_prob(self, value: torch.Tensor):
        """
        Compute log probability of gate token samples.
        
        Args:
            value: Tensor (..., 2k+1) containing [type, indices, signs].
        
        Returns:
            Log probability as sum of type and literal log probabilities.
        """
        t, l = torch.split(value, [1, 2 * self._k], dim=-1)
        t = t.squeeze(-1).float()
        
        logp_t = self.ttype_dist.log_prob(t)
        logp_l = self.literals_dist.log_prob(l)
        
        return logp_t + logp_l


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Test configuration
    import torch.nn.functional as F
    n = 5
    k = 3  # Select top-k elements (subset size)
    zeta = F.softplus(torch.randn(1))  # Random logit for Bernoulli
    phi = torch.randn(n)  # Random logits for 5 elements
    psi = F.softplus(torch.randn(n)) # Random logits for sign selection (positive)
    num_samples = 30000  # Number of samples to generate for testing
    
    # Create the Gumbel Top-K Subset distribution
    dist = GumbelTopKSubsetWithSign(phi, psi, k)

    # Generate samples and compute their properties
    samples = dist.sample((num_samples,))
    log_probs = dist.log_prob(samples)
    entropy = dist.entropy()

    # Display basic results
    print(f"Bernoulli logit (zeta): {zeta.item():.4f}")
    print("Phi (logits):")
    print(phi)
    print("Psi (sign logits):")
    print(psi)
    print(f"Sampling {num_samples} subsets of size {k} from {n} elements")
    print("Samples:")
    print(samples)
    print("Log probabilities:")
    print(log_probs)
    
    # idx, sgn = torch.split(samples, [k, k], dim=-1)
    # literals = (idx + 1) * (1 - 2 * sgn)
    
    # # Count occurrences of each unique subset
    # lookup = defaultdict(int)
    # for row in literals:
    #     s = set(row.tolist())
    #     lookup[frozenset(s)] += 1
    
    # # Compare empirical vs theoretical probabilities
    # print("\nSubset probability comparison:")
    # print("-" * 60)
    # est_entropy = 0
    
    # for subset, count in lookup.items():
    #     # Empirical probability from sampling
    #     est_prob = count / num_samples
    #     est_entropy += - (est_prob * math.log(est_prob))
        
    #     # Theoretical probability from distribution
    #     l = torch.tensor(list(subset), device=dist._device, dtype=torch.long)
    #     s = l.lt(0).int()
    #     l = torch.abs(l) - 1
    #     subset_tensor = torch.cat([l, s], dim=-1)
    #     logp = dist.log_prob(subset_tensor).item()
    #     cal_prob = math.exp(logp)
        
    #     print(f"Subset {set(subset)}: {est_prob:.6f} (est.) - {cal_prob:.6f} (cal.) = {est_prob - cal_prob: .6f}")
    
    # print("Calculated entropy:", entropy.item())
    # print(f"Estimated entropy: {est_entropy:.6f}")
    