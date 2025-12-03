import torch

from model.encoder import Encoder, EncoderLayer


def test_encoder_layer():
    model_size = 256
    dim_k = 64
    dim_v = 64
    num_heads = 8
    sequence_length = 12
    dim_ff = 512
    batch_size = 6

    encoder_layer = EncoderLayer(num_heads, dim_k, dim_v, model_size, dim_ff, 0.2)

    test_inputs = torch.rand((batch_size, sequence_length, model_size))
    mask = torch.zeros((batch_size, 1, sequence_length))

    layer_output = encoder_layer(test_inputs, mask)

    assert layer_output.shape == test_inputs.shape


def test_encoder():
    v_size = 10000
    sequence_length = 24
    model_dim = 256
    dim_k = 64
    dim_v = 64
    num_heads = 8
    dim_ff = 256
    batch_size = 6
    encoder_layers = 6

    encoder = Encoder(v_size, sequence_length, num_heads, dim_k, dim_v, model_dim, dim_ff, encoder_layers, dropout=0.1)

    random_input = torch.randint(0, v_size, (batch_size, sequence_length))
    masks = torch.zeros((batch_size, 1, sequence_length))

    encoder_output = encoder(random_input, masks)

    pars = encoder.named_parameters()
    for par in pars:
        print(par[0])

    assert encoder_output.shape == (batch_size, sequence_length, model_dim)
