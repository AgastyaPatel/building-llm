import torch

def casual_mask(T: int, device: torch.device):
    """
    Returns a bool mask where True means *masked* (disallowed)

    """
    m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    return m.view(1, 1, T, T)