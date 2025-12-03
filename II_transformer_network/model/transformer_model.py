import torch

from model.decoder import Decoder
from model.encoder import Encoder
from model.utils import padding_mask, lookahead_mask


class TransformerModel(torch.nn.Module):
    def __init__(self, size_enc_vocab, size_dec_vocab, seq_length_enc, seq_length_dec, num_heads, dim_k, dim_v,
                 dim_model, dim_feedforward, num_enc_layers, num_dec_layers, dropout=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(size_enc_vocab, seq_length_enc, num_heads, dim_k, dim_v, dim_model, dim_feedforward, 
                               num_enc_layers, dropout)
        self.decoder = Decoder(size_dec_vocab, seq_length_dec, num_heads, dim_k, dim_v, dim_model, dim_feedforward, 
                               num_dec_layers, dropout)

        self.linear_layer = torch.nn.Linear(dim_model, size_dec_vocab)
        self.softmax = torch.nn.Softmax()

    def forward(self, encoder_input, decoder_input, *args, **kwargs):
        # YOUR CODE HERE
        # 1) compute mask masking all pad tokens in the source sequence (for encoder input)
        # 2) compute mask masking all pad tokens in the target sequence (for decoder input)
        # 3) compute mask masking all pad tokens in source sequence (for use in decoder)
        # 4) initialize lookahead mask for decoder (so target tokens are masked when generating output sequence)
        # note: if working on a device other than "cpu", the mask must be assigned to the working device, e.g. by ".to(encoder_input.device)"
        # 5) combine the padding and lookahead mask for the decoder input
        # detailed instruction: first repeat the lookahead mask for each batch element,
        # then combine it with the target padding mask using a maximum operation (torch.maximum)
        # 6) run data through this class' encoder
        # 7) run data through this class' decoder
        # 8) apply last linear layer to decoder outputs (to obtain scores for target vocabulary tokens)

        return output
