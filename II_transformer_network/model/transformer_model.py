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
        source_padding_mask = padding_mask(encoder_input)
        # 2) compute mask masking all pad tokens in the target sequence (for decoder input)
        target_padding_mask = padding_mask(decoder_input)
        # 3) compute mask masking all pad tokens in source sequence (for use in decoder)
        source_mask = source_padding_mask.unsqueeze(1).expand(-1, decoder_input.shape[1], -1)
        # 4) initialize lookahead mask for decoder (so target tokens are masked when generating output sequence)
        # note: if working on a device other than "cpu", the mask must be assigned to the working device, e.g. by ".to(encoder_input.device)"
        look_ahead = lookahead_mask(decoder_input.shape[1]).to(decoder_input.device)
        # 5) combine the padding and lookahead mask for the decoder input
        # detailed instruction: first repeat the lookahead mask for each batch element,
        # then combine it with the target padding mask using a maximum operation (torch.maximum)
        batch_size = decoder_input.shape[0]
        look_ahead_repeated = torch.repeat_interleave(look_ahead.unsqueeze(0), batch_size, 0)
        target_padding_mask_expanded = target_padding_mask.unsqueeze(1).expand(-1, decoder_input.shape[1], -1)
        target_mask = torch.maximum(look_ahead_repeated, target_padding_mask_expanded)
        # 6) run data through this class' encoder
        encoder_output = self.encoder(encoder_input, source_padding_mask.unsqueeze(1))
        # 7) run data through this class' decoder
        decoder_output = self.decoder(decoder_input, encoder_output, target_mask, source_mask)
        # 8) apply last linear layer to decoder outputs (to obtain scores for target vocabulary tokens)
        output = self.linear_layer(decoder_output)
        return output
