import math
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Gumbel, Bernoulli
from deprecation import deprecated

class GumbelTopK(Distribution):
    """
    GumbelTopK distribution for sampling from the top-k elements of a weighted distribution.
    This distribution is parameterized by a set of phi and an integer k, which specifies
    the number of top elements to sample from the weighted distribution.
    """
    arg_constraints = { 
        "_phi": constraints.real_vector,
    }
    has_rsample = False
    
    def __init__(self, phi: torch.Tensor, k: int, statis_freq: int = 1536, validate_args = None):
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
        self._phi = phi
        self._k = k
        self._gumbel = Gumbel(0, 1)
        self._statis_freq = statis_freq
        self._validate_args = validate_args
        
        # Calculate temporary variables
        self._psi = torch.exp(phi)
        self._total_psi = self._psi.sum(dim=-1, keepdim=True)
        self._total_phi = self._phi.sum(dim=-1, keepdim=True)
        
        # Initialize the distribution
        batch_shape = phi.shape[:-1]
        event_shape = torch.Size([k])
        super().__init__(batch_shape, event_shape, validate_args)
        
    def expand(self, batch_shape, _instance=None):
        """
        Returns a new GumbelTopK instance with batch_shape = batch_shape + event_shape,
        where the internal tensors are just views on the original data (no copies).
        """
        # Create a new instance of GumbelTopK (or reuse _instance if provided)
        new = self._get_checked_instance(GumbelTopK, _instance)
        batch_shape = torch.Size(batch_shape)
        
        # Expand the internal phi tensor to the new batch shape
        new._phi = self._phi.expand(*batch_shape, self._phi.size(-1))
        
        # Recompute psi and related cached values for the new batch shape
        new._psi = torch.exp(new._phi)
        new._total_psi = new._psi.sum(dim=-1, keepdim=True)
        
        # Copy over other parameters
        new._gumbel = self._gumbel
        new._k = self._k
        new._statis_freq = self._statis_freq
        
        # Initialize the superclass with the new batch shape and event shape
        super(GumbelTopK, new).__init__(batch_shape, torch.Size([self._k]), self._validate_args)
        return new
    
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
    
    def log_prob(self, value: torch.Tensor):
        """ 
        Calculate the log probability of the given value under the GumbelTopK distribution.
        Args:
            value (torch.Tensor): The indices of the top-k elements to calculate the log probability for.
        Returns:
            torch.Tensor: A tensor containing the log probabilities of the given indices.
        """
        phi = self._phi[(None,) * (value.dim() - self._phi.dim()) + (...,)]
        rshape = value.shape[:-self._phi.dim()] + self._phi.shape  # shape to match value
        phi = phi.expand(*rshape)  # expand phi to match value shape
        
        psi = self._psi[(None,) * (value.dim() - self._psi.dim()) + (...,)]
        rshape = value.shape[:-self._psi.dim()] + self._psi.shape
        psi = psi.expand(*rshape)  # expand psi to match value shape
        
        phivals = phi.gather(-1, value) # [..., k], gather phi values for the top-k indices
        psivals = psi.gather(-1, value) # [..., k], gather psi

        num = phivals.sum(dim=-1) # scalar [...], sum of phi values for the top-k indices

        cumsum = torch.cumsum(psivals, dim=-1) # [..., k]
        excl_cumsum = torch.roll(cumsum, shifts=1, dims=-1)
        excl_cumsum[..., 0] = 0 # [..., k], exclude the first element

        denom = self._total_psi - excl_cumsum # [..., k]
        denom_logs = torch.log(denom).sum(dim=-1) # scalar [...], log of the denominator
        
        return num - denom_logs # scalar [...], log probability of the top-k indices
        
    def entropy(self):
        """
        Estimate the entropy of the GumbelTopK distribution using sampling.
        Returns:
            torch.Tensor: A tensor containing the estimated entropy of the distribution.
        """
        sample_shape = torch.Size([self._statis_freq])
        x = self.log_prob(self.sample(sample_shape))
        
        return -x.mean(dim=0)

@deprecated    
class GumbelTopKSubset(GumbelTopK):
    def __init__(self, *args, **kwargs):
        """
        Initialize the GumbelTopKSubset distribution.
        This is a subclass of GumbelTopK that can be used for specific subsets of the distribution.
        """
        super().__init__(*args, **kwargs)
    
    def log_prob(self, value):
        """
        Calculate the log probability of the given value under the GumbelTopKSubset distribution.
        This method overrides the log_prob method of the parent class.
        Args:
            value (torch.Tensor): The indices of the top-k elements to calculate the log probability for.
        Returns:
            torch.Tensor: A tensor containing the log probabilities of the given indices.
        """
        # Note: The probability of a subset is the sum of the probabilities of all permutations of 
        # the elements in the subset, which is equivalent to the number of all permutations times
        # the mean probability. To estimate this, we adopt importance sampling technique: 
        #       (1) sample a number of random permutations of elements in the subset.
        #       (2) calculate the mean probability.
        #       (3) multiply it by the number of permutations, which is exactly the factorial of the size 
        #           of the subset.
        #       (4) return the log of that value.
        # Caution: This is an approximation and may not be exact, and if the sampling frequency is low, 
        # the result may be highly inaccurate (and could be even over 1).
        # TODO: since the result could be over 1, apply softmax to the result to make it a valid probability.
        
        perms = torch.rand(self._statis_freq, *value.shape, device=value.device).argsort(dim=-1) # random sample to get the indices
        value = value.expand(self._statis_freq, *value.shape) # expand value to match the sample shape
        value = value.gather(-1, perms) # gather the values based on the random indices
        logp = super().log_prob(value) # calculate log probabilities using the parent class method
        m = math.factorial(value.shape[-1]) # factorial of the number of top-k elements
        p = torch.exp(logp) # exponentiate the log probabilities
        
        return torch.log(p.mean(dim=0) * m) # return the mean log probability multiplied by the factorial

class GateTokenDistribution(Distribution):
    """
    GateTokenDistribution is a specialized distribution for sampling gate tokens.
    It inherits from GumbelTopK and is used to sample tokens based on their probabilities.
    """
    def __init__(self, phi: torch.Tensor, p_EOS: torch.Tensor, k: int, statis_freq: int = 1536, validate_args = None):
        """
        Initialize the GateTokenDistribution.
        Args:
            phi (torch.Tensor): A tensor of shape (..., n) representing the weights for the distribution.
            p_EOS (torch.Tensor): A tensor of shape (..., 1) representing the probability of EOS token.
            k (int): The number of top elements to sample.
            statis_freq (int): Number of samples to use for estimating entropy.
            validate_args (bool): Whether to validate arguments.
        """
        if not isinstance(p_EOS, torch.Tensor):
            raise TypeError("`p_EOS` must be torch.Tensor")
        if p_EOS.dim() != 1:
            raise ValueError("`p_EOS` must be 1-dimensional")
        if p_EOS.shape[:-1] != phi.shape[:-1]:
            raise ValueError("`p_EOS` must have the same batch shape as `phi`")
        
        self.bernoulli = Bernoulli(p_EOS)
        self.gumbel_topk = GumbelTopK(phi, k, statis_freq, validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the GateTokenDistribution.
        Args:
            sample_shape (torch.Size): The shape of the samples to draw.
        Returns:
            torch.Tensor: A tensor containing the indices of the top-k elements sampled from the distribution.
        """
        # Call the parent class sample method to get the top-k indices
        eos = self.bernoulli.sample(sample_shape)
        topk_indices = self.gumbel_topk.sample(sample_shape)
        
        return torch.cat([topk_indices, eos.unsqueeze(-1)], dim=-1)  # concatenate EOS token at the end

    def log_prob(self, value: torch.Tensor):
        """
        Calculate the log probability of the given value under the GateTokenDistribution.
        Args:
            value (torch.Tensor): The indices of the top-k elements to calculate the log probability for.
        Returns:
            torch.Tensor: A tensor containing the log probabilities of the given indices.
        """
        # Call the parent class log_prob method to get the log probabilities
        return super().log_prob(value) * (1 - value[..., -1]) + self.bernoulli.log_prob(value[..., -1])

if __name__ == "__main__":
    # Example usage
    phi = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1]])
    k = 2
    gumbel_topk_dist = GumbelTopK(phi, k, statis_freq=2000)
    
    samples = gumbel_topk_dist.sample()
    print("Samples:", samples)
    log_probs = gumbel_topk_dist.log_prob(samples)
    print("Log Probabilities:", log_probs)
    entropy = gumbel_topk_dist.entropy()
    print("Entropy:", entropy)
    # You can check the entropy values by hand. The answer should be [2.4764, 2.4849].
    # Note that 2.4849 is exactly log(12), which is the maximum entropy for a 
    # uniform distribution over 4 * 3 = 12 sequences.
    
    new_g = gumbel_topk_dist.expand(torch.Size([2]))
    print("Expanded Samples:", new_g.sample())
    print("Expanded Log Probabilities:", new_g.log_prob(samples))
    print("Expanded Entropy:", new_g.entropy())