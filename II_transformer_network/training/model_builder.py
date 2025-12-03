import torch

from model.model_config import ModelConfig
from model.transformer_model import TransformerModel


def build_model_for_data(source_seq_length, target_seq_length, source_vocab, target_vocab, model_config: ModelConfig, dropout=0):
    source_v_size = len(source_vocab)
    target_v_size = len(target_vocab)

    model = TransformerModel(source_v_size, target_v_size, source_seq_length, target_seq_length,
                             model_config.num_heads, model_config.dim_keys, model_config.dim_values,
                             model_config.dim_model, model_config.dim_feedforward, model_config.num_encoder_layers,
                             model_config.num_decoder_layers, dropout)

    return model


def load_model_weights(model: TransformerModel, model_state_dict_file_path: str):
    state_dict = torch.load(model_state_dict_file_path)
    model.load_state_dict(state_dict)
