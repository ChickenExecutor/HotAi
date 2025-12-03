import torch
from torch.nn import Linear

from model.dot_product_attention import DotProductAttention


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, num_heads, dim_q, dim_k, dim_v, dim_model, *args, **kwargs):
        # computes multi-head attention for the transformer model,
        # i.e. tensors query, keys, values are input and projected using dedicated learnable linear layers,
        # subsequently, num_heads individual attention computations are performed
        # the attentions results of all heads are concatenated to a single tensor
        super().__init__(*args, **kwargs)
        self.attention = DotProductAttention()
        self.num_heads = num_heads
        self.proj_q = Linear(dim_model, dim_q)
        self.proj_k = Linear(dim_model, dim_k)
        self.proj_v = Linear(dim_model, dim_v)
        self.proj_out = Linear(dim_v, dim_model)


    def _expand_heads_and_reorder(self, tensor):
        # expands an additional dimension to a tensor representing the number of attention heads
        # takes a tensor of shape (x_1, x_2, x_3 = num_heads * z) and reshapes it to (x_1, num_heads, x_2, z)
        # the reshaped tensor can be fed to the class' DotProductAttention-instance
        # to efficiently compute all attention heads simultaneously
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], self.num_heads, -1))
        tensor = torch.permute(tensor, (0, 2, 1, 3))
        return tensor

    def _concat_heads_and_reorder(self, tensor):
        # reverses the method _expand_heads_and_reorder
        # takes a tensor of shape (x_1, num_heads, x_2, z) and reshapes it to (x_1, x_2, x_3 = num_heads * z)
        # once attention is computed, the results can be reshaped concatenating the individual heads back together
        # to obtain overall results consistent in dimension to the input module's input
        tensor = torch.permute(tensor, (0, 2, 1, 3))
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor

    def forward(self, queries, keys, values, mask=None):
        # queries: 3-dim. tensor of shape (batch_size, sequence_length_q, dim_model)
        # keys: 3-dim. tensor of shape (batch_size, sequence_length_k, dim_model)
        # values: 3-dim. tensor of shape (batch_size, sequence_length_k, dim_model)
        # mask: 3-dim. tensor of shape (batch_size, sequence_length_q, sequence_length_k),
        #  or shape (batch_size, 1, sequence_length_k) if the same padding mask is applied at every position

        # expand mask for multihead attention: add additional dimension for multiple heads
        # (use identical mask for each head)
        if mask is not None:
            mask_multihead = torch.repeat_interleave(mask.unsqueeze(1), self.num_heads, 1)
        else:
            mask_multihead = None

        # YOUR CODE HERE
        # 1) Project queries, keys and values using respective linear projection layers
        # 2) Use method _expand_heads_and_reorder to reorganize resulting tensors for multihead attention computation
        # 3) Compute multihead attention
        # 4) Use method _concat_heads_and_reorder to concatenate the attention heads back together
        # 5) Apply output projection linear layer to attention results

        return output
