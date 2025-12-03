import torch

from model.transformer_model import TransformerModel

def test_transformer():
    model_size = 512
    dim_k = 64
    dim_v = 64
    num_heads = 8
    sequence_length_in = 12
    sequence_length_out = 8
    dim_ff = 2048
    batch_size = 24
    voc_size = 1000

    transformer_model = TransformerModel(voc_size, voc_size, sequence_length_in, sequence_length_out, num_heads,
                                         dim_k, dim_v, model_size, dim_ff, num_enc_layers=6, num_dec_layers=6,
                                         dropout=0.1)

    encoder_input = torch.randint(0, voc_size, (batch_size, sequence_length_in))
    decoder_output = torch.randint(0, voc_size, (batch_size, sequence_length_out))

    transformer_output = transformer_model(encoder_input, decoder_output)

    assert transformer_output.shape == (batch_size, sequence_length_out, voc_size)