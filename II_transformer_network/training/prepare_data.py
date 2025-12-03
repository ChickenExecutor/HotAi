import csv
import pickle as pcl
from typing import List, Dict

import numpy as np
import torchtext

from training import *


def get_vocab_dict(tokenized_sentences: List[List[str]]) -> Dict[str, int]:
    vocab = set()
    token_dict = {pad_token: 0}
    vocab.add(eos_token)
    vocab.add(start_token)
    for token_list in tokenized_sentences:
        vocab = vocab.union(token_list)

    for i, word in enumerate(sorted(vocab)):
        token_dict[word] = i + 1

    return token_dict


def preprocess_data(sentences: List[str], tokenizer) -> (Dict[str, int], np.ndarray):
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.replace("„", "\"")
        sentence = sentence.replace("“", "\"")
        sentence = sentence.replace("‟", "\"")
        sentence = sentence.replace("”", "\"")
        sentence = sentence.replace("‚", ",")
        tokenized_sentences.append(tokenizer(sentence))
    vocab_dict = get_vocab_dict(tokenized_sentences)

    sequence_length = np.array([len(token_list) for token_list in tokenized_sentences]).max()

    data = [[vocab_dict[token] for token in sentence] for sentence in tokenized_sentences]

    for entry in data:
        while len(entry) < sequence_length:
            entry.append(0)

    return vocab_dict, np.array(data)


def load_and_process_raw_data():
    data = csv.reader(open(get_existing_data_file_path(), "r", encoding="utf-8"), delimiter=",")

    data_np = []
    for row in data:
        data_np.append([start_token + " " + row[0] + " " + eos_token,
                        start_token + " " + row[1] + " " + eos_token])
    data_np = np.array(data_np)

    source_dict, source_data = preprocess_data(data_np[:, 0], torchtext.data.get_tokenizer("basic_english"))
    target_dict, target_data = preprocess_data(data_np[:, 1], torchtext.data.get_tokenizer("basic_english"))

    return source_data, target_data, source_dict, target_dict


def store_data(source_data, target_data, source_dict, target_dict, output_file):
    data_dict = {
        "source_data": source_data,
        "target_data": target_data,
        "source_dict": source_dict,
        "target_dict": target_dict
    }
    pcl.dump(data_dict, open(output_file, "wb"))


def restore_data(data_file):
    data = pcl.load(open(data_file, "rb"))
    source_data = data["source_data"]
    target_data = data["target_data"]
    source_dict = data["source_dict"]
    target_dict = data["target_dict"]
    return source_data, target_data, source_dict, target_dict


if __name__ == '__main__':
    source_data, target_data, source_vocab, target_vocab = load_and_process_raw_data()
    np.random.seed(123)
    data_indices = np.arange(len(source_data))
    np.random.shuffle(data_indices)
    source_data = source_data[data_indices]
    target_data = target_data[data_indices]
    store_data(source_data, target_data, source_vocab, target_vocab, preprocessed_data_file_write_path)
