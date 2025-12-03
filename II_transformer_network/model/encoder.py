import torch

from model.multi_head_attention import MultiHeadAttention
from model.position_embedding import PositionEmbeddingLayer


class EncoderLayer(torch.nn.Module):
    def __init__(self, num_heads, dim_k, dim_v, dim_model, dim_feedforward, dropout=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihead_attention = MultiHeadAttention(num_heads, dim_k, dim_k, dim_v, dim_model)

        self.linear_1 = torch.nn.Linear(dim_model, dim_feedforward)
        self.linear_2 = torch.nn.Linear(dim_feedforward, dim_model)
        self.norm_1 = torch.nn.LayerNorm(dim_model)
        self.norm_2 = torch.nn.LayerNorm(dim_model)
        self.activation_1 = torch.nn.ReLU()
        self.activation_2 = torch.nn.ReLU()
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.dropout_3 = torch.nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        # YOUR CODE HERE
        # 1) compute multihead self-attention
        # 2) apply dropout
        # 3) implement add & norm
        # 4) Feed forward part: -> linear layer -> activation -> dropout -> linear layer -> activation -> dropout
        # 5) add & norm
        return layer_output


class Encoder(torch.nn.Module):

    def __init__(self, v_size, sequence_length, num_heads, dim_k, dim_v, dim_model, dim_ff, n, dropout=0, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.position_encoding = PositionEmbeddingLayer(sequence_length, v_size, dim_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.encoder_layers = [EncoderLayer(num_heads, dim_k, dim_v, dim_model, dim_ff, dropout) for i in range(n)]
        [self.register_module(f"encoder_layer_{i}", layer) for i, layer in enumerate(self.encoder_layers)]

    def forward(self, input_sequence, padding_masks):
        encoded_input = self.position_encoding(input_sequence)

        x = self.dropout(encoded_input)

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, padding_masks)

        return x
