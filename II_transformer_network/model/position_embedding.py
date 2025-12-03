import torch


class PositionEmbeddingLayer(torch.nn.Module):

    def __init__(self, sequence_length, vocab_size, dim_output, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_embedding = torch.nn.Embedding(vocab_size, dim_output)
        self.position_embedding = torch.nn.Embedding(sequence_length, dim_output)

    def forward(self, input_sequence):
        # computes embeddings for an input sequence sa the sum of (a) token embeddings and (b) positional embeddings
        # (a) token embeddings are computed based on input sequence using "self.word_embedding"
        # (b) position embeddings are computed based on positional indices using "self.position_embedding"
        # for the attentive reader:
        # contrary to the original transformer architecture positional embeddings are learnable in this implementation
        position_indices = torch.arange(0, input_sequence.shape[-1], device=input_sequence.device)
        embeddings_positions = self.position_embedding(position_indices)
        embeddings_tokens = self.word_embedding(input_sequence)
        return embeddings_tokens + embeddings_positions
