import torch


def padding_mask(input: torch.Tensor) -> torch.Tensor:
    mask = torch.where(input == torch.zeros_like(input), torch.ones_like(input), torch.zeros_like(input))
    return mask


def lookahead_mask(shape: int):
    mask = torch.triu(torch.ones(shape, shape), diagonal=1)
    return mask
