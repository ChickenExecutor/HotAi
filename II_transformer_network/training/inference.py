import os

import torch

from model.model_config import ModelConfig
from training import eos_token, start_token, get_training_dirs, get_model_state_dict_files, \
    config_file_name, get_existing_preprocessed_data_file
from training.model_builder import build_model_for_data, load_model_weights
from training.prepare_data import restore_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data_and_model(model_config_file, model_state_dict_file):
    model_config = ModelConfig.read_from_file(model_config_file)
    source_data, target_data, source_vocab, target_vocab = restore_data(get_existing_preprocessed_data_file())
    model = build_model_for_data(source_data.shape[1], target_data.shape[1], source_vocab, target_vocab, model_config)
    model.to(device)
    load_model_weights(model, model_state_dict_file)

    return source_data, target_data, source_vocab, target_vocab, model


def decipher_tokens(token_list, vocab):
    words = []
    for token in token_list:
        for key, value in vocab.items():
            if token == value:
                words.append(key)
                if key == eos_token:
                    return words
                continue
    return words


def run_model(model, source_data, source_vocab, target_vocab, max_generated_text_length=100, number_of_runs=10,
              target_data=None, device=device):
    for i in range(min(number_of_runs, source_data.shape[0])):
        source_tokens = torch.tensor(source_data[i], device=device).long()
        generated_output = torch.zeros([1]).long()
        generated_output[0] = target_vocab[start_token]
        generated_output = generated_output.to(device)

        input_text = decipher_tokens(source_tokens, source_vocab)
        input_text = " ".join(input_text)
        print(f"Input: {input_text}")

        for k in range(max_generated_text_length):
            output_probs = model.forward(source_tokens.unsqueeze(0), generated_output.unsqueeze(0))
            next_token = torch.Tensor.argmax(output_probs[0, -1])
            generated_output = torch.concatenate([generated_output, next_token.unsqueeze(0)], dim=-1)
            if next_token == target_vocab[eos_token]:
                break

        output_text = decipher_tokens(generated_output, target_vocab)

        output_text = " ".join(output_text)
        print(f" Output: {output_text}")

        if target_data is not None:
            target_text = decipher_tokens(target_data[i], target_vocab)
            target_text = " ".join(target_text)
            print(f" Target was '{target_text}'.")


def load_and_run(model_config_file, model_state_dict_file, number_of_runs=10):
    source_data, target_data, source_vocab, target_vocab, model = load_data_and_model(model_config_file,
                                                                                      model_state_dict_file)
    model.train(False)
    run_model(model, source_data, source_vocab, target_vocab,
              number_of_runs=number_of_runs, target_data=target_data, device=device)


def run_on_string(model, input_string, source_vocab, target_vocab):
    input_words = input_string.split()
    input_tokens = [source_vocab[start_token]]
    for word in input_words:
        if word in source_vocab:
            input_tokens.append(source_vocab[word])
        else:
            input_tokens.append(0)
    input_tokens.append(source_vocab[eos_token])
    input_tokens = torch.Tensor(input_tokens).unsqueeze(0)

    run_model(model, input_tokens, source_vocab, target_vocab, device=device)


def find_last_training_and_run(number_of_runs=10):
    training_dirs = get_training_dirs()
    if len(training_dirs) > 0:
        training_dir = sorted(training_dirs)[-1]
        model_config_file = os.path.join(training_dir, config_file_name)
        model_state_dict_files = get_model_state_dict_files(training_dir)
        epoch_no = max(model_state_dict_files.keys())
        model_state_dict_file = model_state_dict_files[epoch_no]
        print("Inferencing samples with model from {}".format(model_state_dict_file))
        load_and_run(model_config_file, model_state_dict_file, number_of_runs)


if __name__ == '__main__':
    find_last_training_and_run(10)
