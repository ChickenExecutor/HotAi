import torch

from model import decoder


def test_decoder_layer():
    model_size = 512
    dim_k = 64
    dim_v = 64
    num_heads = 8
    sequence_length = 512
    dim_ff = 2048
    batch_size = 24
    voc_size = 1000

    decoder_layer = decoder.DecoderLayer(num_heads, dim_k, dim_v, model_size, dim_ff, 0.2)

    output_sequence = torch.rand((batch_size, sequence_length, model_size))
    test_inputs = torch.rand((batch_size, sequence_length, model_size))
    mask = torch.zeros((batch_size, 1, sequence_length))

    layer_output = decoder_layer(output_sequence, test_inputs, mask, None)

    assert layer_output.shape == test_inputs.shape


def test_decoder():
    model_size = 256
    dim_k = 64
    dim_v = 64
    num_heads = 8
    sequence_length = 12
    dim_ff = 512
    batch_size = 6
    voc_size = 1000

    decoder_inst = decoder.Decoder(voc_size, sequence_length, num_heads, dim_k, dim_v, model_size, dim_ff, 6)

    output_sequence = torch.randint(0, voc_size, (batch_size, sequence_length))
    masks = torch.zeros((batch_size, 1, sequence_length))

    encoder_output = torch.rand((batch_size, sequence_length, model_size))

    output_sequence = decoder_inst(output_sequence, encoder_output, masks, None)
    assert output_sequence.shape == (batch_size, sequence_length, model_size)
