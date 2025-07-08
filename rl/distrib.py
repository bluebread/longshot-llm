import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Gumbel

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
    
    def __init__(self, phi: torch.Tensor, k: int, num_sample_entropy: int = 1536, validate_args = None):
        """
        Initialize the GumbelTopK distribution.
        Args:
            phi (torch.Tensor): A tensor of shape (..., n) representing the weights for the distribution.
            k (int): The number of top elements to sample.
            num_sample_entropy (int): Number of samples to use for estimating entropy.
            validate_args (bool): Whether to validate arguments.
        """
        if not isinstance(phi, torch.Tensor) or not isinstance(k, int):
            raise TypeError("`phi` must be torch.Tensor, and `k` must be int")
        if phi.dim() < 1:
            raise ValueError("`phi` must be at least 1-dimensional")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("`k` must be a positive integer (>=1)")
        if not isinstance(num_sample_entropy, int) or num_sample_entropy <= 0:
            raise ValueError("`num_sample_entropy` must be a positive integer (>=1)")
        
        # Initialize parameters
        self._phi = phi
        self._k = k
        self._gumbel = Gumbel(0, 1)
        self._num_sample_entropy = num_sample_entropy
        
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
        new._k = torch.Size([self._k])
        new._num_sample_entropy = self._num_sample_entropy
        
        # Initialize the superclass with the new batch shape and event shape
        super(GumbelTopK, new).__init__(batch_shape, torch.Size([self._k]), self.validate_args)
        return new
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the GumbelTopK distribution.
        Args:
            sample_shape (torch.Size): The shape of the samples to draw.
        Returns:
            torch.Tensor: A tensor containing the indices of the top-k elements sampled from the GumbelTopK distribution.
        """
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
        rshape = value.shape[:-self._phi.dim()] + (1,) * self._phi.dim()
        phi = phi.repeat(*rshape)  # repeat phi to match value shape
        
        psi = self._psi[(None,) * (value.dim() - self._psi.dim()) + (...,)]
        rshape = value.shape[:-self._psi.dim()] + (1,) * self._psi.dim()
        psi = psi.repeat(*rshape)  # repeat psi to match value shape
        
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
        sample_shape = torch.Size([self._num_sample_entropy])
        x = self.log_prob(self.sample(sample_shape))
        
        return -x.mean(dim=0)
     
if __name__ == "__main__":
    # Example usage
    phi = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    k = 2
    gumbel_topk_dist = GumbelTopK(phi, k)
    
    samples = gumbel_topk_dist.sample()
    print("Samples:", samples)
    log_probs = gumbel_topk_dist.log_prob(samples)
    print("Log Probabilities:", log_probs)
    entropy = gumbel_topk_dist.entropy()
    print("Entropy:", entropy)
    