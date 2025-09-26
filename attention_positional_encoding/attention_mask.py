import torch

def casual_mask(T: int, device: torch.device=torch.device("cpu")):
    """
    Returns a bool mask where True means *masked* (disallowed)

    """
    m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    return m.view(1, 1, T, T)