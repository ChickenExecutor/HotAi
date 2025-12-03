### STUDENT NAME:
### STUDENT IDENTIFIER (U****):
### COLLABORATION WITH:

import gc
import os

import numpy as np
import numpy.random
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text.perplexity import Perplexity
from tqdm import tqdm

from model.model_config import ModelConfig
from model.transformer_model import TransformerModel
from training import (training_dir, model_state_dict_file_name, config_file_name,
                      preprocessed_data_file_write_path, get_existing_preprocessed_data_file)
from training.inference import run_model, load_model_weights
from training.model_builder import build_model_for_data
from training.prepare_data import load_and_process_raw_data, store_data, restore_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Split config
train_ratio = 0.8

# Set dropout rate
dropout = 0.3

# Configure training (adjust depending on available GPU memory)
batch_size = 1024
num_epochs = 20

# Large model version: trains several hours on GPU
model_config_large = ModelConfig(num_heads=8,
                                 dim_keys=64,
                                 dim_values=64,
                                 dim_model=512,
                                 dim_feedforward=1024,
                                 num_encoder_layers=4,
                                 num_decoder_layers=4)

# Small model version: trains approx. one hour on GPU
model_config_small = ModelConfig(num_heads=2,
                                 dim_keys=16,
                                 dim_values=16,
                                 dim_model=32,
                                 dim_feedforward=64,
                                 num_encoder_layers=1,
                                 num_decoder_layers=1)

# Use either version - or build your own model!
# model_config = model_config_large
model_config = model_config_small


def run_on_batch(model: TransformerModel, data_x: torch.LongTensor, data_y: torch.LongTensor, loss_fn,
                 ppl: Perplexity, train: bool):
    decoder_input = data_y[:, :-1]
    decoder_output = data_y[:, 1:]

    model.train(train)
    predictions = model(data_x, decoder_input)

    if ppl is not None:
        ppl.update(predictions, decoder_output)

    predictions_stacked = predictions.reshape([-1, predictions.shape[-1]])
    target_stacked = decoder_output.reshape([-1])

    # pad_prediction = torch.zeros((predictions.shape[-1]), device=device)
    # pad_prediction[0] = 1e3

    # predictions_stacked = torch.where(target_stacked.unsqueeze(1) != 0, predictions_stacked, pad_prediction)

    loss = loss_fn(predictions_stacked, target_stacked)

    return loss


def run_epoch(model: TransformerModel, optimizer: torch.optim.Optimizer, data_loader: DataLoader,
              loss_fn, ppl_metric: Perplexity, train: bool):
    complete_loss = 0
    train_data_tqdm = tqdm(data_loader, "Train" if train else "Test", total=len(data_loader), unit="batch")
    for data_x, data_y in train_data_tqdm:
        loss = run_on_batch(model, data_x, data_y, loss_fn, ppl_metric, train)
        complete_loss += float(loss)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del data_x, data_y, loss
        clear_memory()

    ppl_value = ppl_metric.compute()
    ppl_metric.reset()

    return complete_loss, ppl_value


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def get_datasets(train_ratio, preprocess_data: bool = False):
    if preprocess_data:
        source_data, target_data, source_vocab, target_vocab = load_and_process_raw_data()
        numpy.random.seed(123)
        data_indices = np.arange(len(source_data))
        numpy.random.shuffle(data_indices)
        source_data = source_data[data_indices]
        target_data = target_data[data_indices]
        store_data(source_data, target_data, source_vocab, target_vocab, preprocessed_data_file_write_path)
    else:
        source_data, target_data, source_vocab, target_vocab = restore_data(get_existing_preprocessed_data_file())

    amount_train = int(train_ratio * len(source_data))
    data_train = TensorDataset(torch.tensor(source_data[:amount_train], device=device),
                               torch.tensor(target_data[:amount_train], device=device).long())
    data_val = TensorDataset(torch.tensor(source_data[amount_train:], device=device),
                             torch.tensor(target_data[amount_train:], device=device).long())

    return data_train, data_val, source_vocab, target_vocab


def train(device, preprocess_data: bool = True, training_dir: str = training_dir, start_epoch: int = 0):
    data_train, data_val, source_vocab, target_vocab = get_datasets(train_ratio, preprocess_data)
    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    amount_train, source_seq_length = data_train.tensors[0].shape
    amount_val, target_seq_length = data_val.tensors[1].shape

    train_record_writer = SummaryWriter(log_dir=training_dir)
    model_state_dict_file = os.path.join(training_dir, model_state_dict_file_name)

    model = build_model_for_data(source_seq_length, target_seq_length, source_vocab, target_vocab,
                                 model_config, dropout)

    load_model = start_epoch > 0
    if load_model:
        load_model_weights(model, model_state_dict_file.format(start_epoch))

    model_config.write_to_file(os.path.join(training_dir, config_file_name))

    model.to(device)

    opt = torch.optim.RMSprop(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=0)
    ppl_metric = Perplexity(ignore_index=0).to(device)

    for current_epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_loss, train_ppl = run_epoch(model, opt, data_loader_train, loss_fn, ppl_metric, train=True)
        train_loss /= amount_train

        train_record_writer.add_scalar("Loss/train", train_loss, current_epoch + 1)
        train_record_writer.add_scalar("Ppl/train", train_ppl, current_epoch + 1)

        val_loss, val_ppl = run_epoch(model, opt, data_loader_val, loss_fn, ppl_metric, train=False)
        val_loss /= amount_val

        train_record_writer.add_scalar("Loss/val", val_loss, current_epoch + 1)
        train_record_writer.add_scalar("Ppl/val", val_ppl, current_epoch + 1)

        print(f"Epoch: {current_epoch + 1:>5}      Loss:   {train_loss:.5f}      Val Loss:   {val_loss:.5f}")

        torch.save(model.state_dict(), model_state_dict_file.format(current_epoch))
        train_record_writer.flush()

    run_model(model, data_val.tensors[0], source_vocab, target_vocab, device=device)


if __name__ == '__main__':
    train(device, preprocess_data=False, start_epoch=0)
