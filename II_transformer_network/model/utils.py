import torch


def padding_mask(input: torch.Tensor) -> torch.Tensor:
    # create a mask based on padding (0) tokens in the data
    mask = torch.where(input == torch.zeros_like(input), torch.ones_like(input), torch.zeros_like(input))
    return mask


def lookahead_mask(shape: int):
    # create a lookahead mask for the decoder (so decoder can not use expected output for its predictions)
    mask = torch.triu(torch.ones(shape, shape), diagonal=1)
    return mask
