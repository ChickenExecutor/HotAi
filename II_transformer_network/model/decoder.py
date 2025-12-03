import torch

from model.multi_head_attention import MultiHeadAttention
from model.position_embedding import PositionEmbeddingLayer


class DecoderLayer(torch.nn.Module):

    def __init__(self, num_heads, dim_k, dim_v, dim_model, dim_feedforward, dropout=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihead_attention_1 = MultiHeadAttention(num_heads, dim_k, dim_k, dim_v, dim_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.norm_1 = torch.nn.LayerNorm(dim_model)
        self.multihead_attention_2 = MultiHeadAttention(num_heads, dim_k, dim_k, dim_v, dim_model)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.norm_2 = torch.nn.LayerNorm(dim_model)
        self.linear_1 = torch.nn.Linear(dim_model, dim_feedforward)
        self.linear_2 = torch.nn.Linear(dim_feedforward, dim_model)
        self.activation_1 = torch.nn.ReLU()
        self.activation_2 = torch.nn.ReLU()
        self.dropout_3 = torch.nn.Dropout(dropout)
        self.dropout_4 = torch.nn.Dropout(dropout)
        self.norm_3 = torch.nn.LayerNorm(dim_model)

    def forward(self, x, encoder_output, target_mask, source_mask):
        # YOUR CODE HERE
        # 1) apply masked multihead self-attention
        # 2) dropout
        # 3) add & norm
        # 4) apply masked multihead attention on encoder output (use encoder output as keys, values)
        # 5) dropout
        # 6) add & norm
        # 7) implement feed forward part of encoder layer: linear layer -> activation -> dropout -> linear layer -> activation -> dropout
        # 8) add & norm

        return output


class Decoder(torch.nn.Module):

    def __init__(self, v_size, sequence_length, num_heads, dim_k, dim_v, dim_model, dim_ff, n, dropout=0, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.position_encoding = PositionEmbeddingLayer(sequence_length, v_size, dim_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.decoder_layers = [DecoderLayer(num_heads, dim_k, dim_v, dim_model, dim_ff, dropout) for i in range(n)]
        [self.register_module(f"decoder_layer_{i}", layer) for i, layer in enumerate(self.decoder_layers)]

    def forward(self, outputs, encoder_output, target_sequence_mask, source_sequence_mask):
        encoded_outputs = self.position_encoding(outputs)

        x = self.dropout(encoded_outputs)

        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, encoder_output, target_sequence_mask, source_sequence_mask)

        return x
