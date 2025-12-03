import math

import torch


class DotProductAttention:
    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask=None) -> torch.Tensor:
        # Compute dot product attention.
        # queries: multidimensional tensor of shape (..., sequence_length_q, dim_q)
        # keys: multidimensional tensor of shape (..., sequence_length_k, dim_k=dim_q)
        # values: multidimensional tensor of shape (..., sequence_length_k, dim_v)
        # mask: multidimensional tensor of shape (..., sequence_length_q, sequence_length_k)
        #   or (..., 1, sequence_length_k) if identical mask is used at every position
        #   Can be used to compute masked attention.
        #   Indicates which sequence items may attend to which parts of the sequence
        #   (will assign zero weight to masked-out relations)
        # Returns weighted values as tensor of shape (..., sequence_length, dim_v)

        dim_q = queries.shape[-1]
        scores = torch.matmul(queries, torch.transpose(keys, -1, -2)) / math.sqrt(dim_q)

        if mask is not None:
            assert mask.shape[:-2] == queries.shape[:-2]
            # set scores to -inf at masked-out sequence positions
            scores = scores.masked_fill(mask == 1, -1e6)

        # YOUR CODE HERE
        # 1) Apply softmax function to obtain weights for values (summing to 1.0)
        # 2) Computed weighted sum of the values according to computed weights

        return weighted_values
