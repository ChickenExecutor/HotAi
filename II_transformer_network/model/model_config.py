import dataclasses
import json


@dataclasses.dataclass
class ModelConfig:
    num_heads: int
    dim_keys: int
    dim_values: int
    dim_model: int
    dim_feedforward: int
    num_encoder_layers: int
    num_decoder_layers: int

    def write_to_file(self, output_file_path: str):
        with open(output_file_path, 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def read_from_file(cls, input_file_path: str):
        with open(input_file_path, 'r') as f:
            config_json = json.load(f)
            return ModelConfig(**config_json)
