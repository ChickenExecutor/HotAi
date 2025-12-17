import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.position_embedding import PositionEmbeddingLayer

def test_embedding_layer():
    v_size = 10000
    sequence_length = 512
    model_dim = 512

    random_input = torch.randint(0, v_size, (10, sequence_length))

    embedding_layer = PositionEmbeddingLayer(sequence_length, v_size, model_dim)

    embeddings = embedding_layer(random_input)
    assert embeddings.shape == (10, sequence_length, model_dim)
