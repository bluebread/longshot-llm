import torch

def random_formula_permutations(
    num_vars: int, 
    num_perms: int, 
    keep_origin: bool = True,
    device: str | torch.device | None = None
    ) -> torch.Tensor:
    """
    Generates a random permutation of variables for a given number of variables and permutations.
    Args:
        num_vars (int): The number of variables to permute.
        num_perms (int): The number of permutations to generate.
        keep_origin (bool): If True, includes the original order as the first permutation. Defaults to True.
        device (str | torch.device | None): The device on which to create the tensor. Defaults to None, which uses the CPU.
    Returns:
        torch.Tensor: A tensor containing the generated permutations.
    """
    device = device if device is not None else torch.device('cpu')
    n = num_vars
    p = num_perms
    s = 1 if keep_origin else 0
    perms = []
    
    if keep_origin:
        perms = [torch.arange(2*n, dtype=torch.int32)]

    for i in range(s,p):
        perm = torch.randperm(n, dtype=torch.int32).repeat(2)
        sgn = torch.randint(0, 2, (n,), dtype=torch.int32)
        perm[:n] += (2*i + sgn) * n
        perm[n:] += (2*i + 1 - sgn) * n
        perms.append(perm)
        
    return torch.cat(perms, dim=0).to(device)

def permute_tensor(
    tensor: torch.Tensor, 
    permutations: torch.Tensor, 
    ) -> torch.Tensor:
    """
    Applies a set of permutations to each row (of last dimension) in a tensor.
    Args:
        tensor (torch.Tensor): The tensor to permute.
        permutations (torch.Tensor): The permutations to apply.
        num_vars (int): The number of variables in the tensor.
    Returns:
        torch.Tensor: The permuted tensor.
    """
    d = tensor.shape[-1]  # Assuming tensor shape is (*, d)
    p = permutations.shape[0] // d
    k = tensor.numel() // d
    r = [1 for _ in range(tensor.ndim - 1)] + [p]
    s = []
    
    for i in range(p):
        s.extend([j * p + i for j in range(k)])
    
    x = tensor.repeat(r) # Repeat the tensor for each permutation # (*, d) -> (*, p*d)
    x = x.index_select(tensor.ndim - 1, permutations).view(-1, d)  # (k, d)
    x = x[s,:].view(p, *tensor.shape)  # Reshape to the original number of permutations
    
    return x

if __name__ == "__main__":
    # Example usage
    num_vars = 2
    num_perms = 3
    device = torch.device('cpu')
    
    permutations = random_formula_permutations(num_vars, num_perms, device)
    print("Generated Permutations:")
    print(permutations)
    
    # Create a sample tensor to permute
    tensor = torch.randn((num_perms, 2 * num_vars), device=device)
    permuted_tensor = permute_tensor(tensor, permutations)
    
    print("Original Tensor:")
    print(tensor)
    print("Permuted Tensor:")
    print(permuted_tensor)