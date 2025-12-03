# Bonus Assignment: Transformer Implementation

In this assignment, you will be implementing your own **transformer model**! :)

Once again, the project structures and some parts of the model implementation are already given.
Nevertheless, there are a number of dedicated functions, methods and code snippets that you will need to implement.
A description of all the tasks you need to complete before you can train your model can be found below.

Once the model is completed, it can be trained for **english-to-german text translation**.
Note, that if you do not have access to a gpu unit, training the model to convergence might not be feasible.
Thus, a complete model training is not a mandatory part of this assignment. 

*The assignment was created and tested using Python 3.12.*

## Information for Work on bwJupyter

If you are working on bwJupyter,
the assignment materials will be copied to your home directory on first profile startup
(look for the 'transformer_network' directory).

On bwJupyter, all required dependencies are already installed in the provided Python environment.
There is a GPU available for training the model.

## The Repository

The given repository is organized as follows:

```
├───checks_and_tests
│   ├───test_decoder.py
│   ├───test_dot_product_attention.py
│   ├───test_embedding.py
│   ├───test_multihead_attention.py
│   ├───test_transformer.py
│   └───test_utils.py
├───data
│   └───deu.txt
├───model
│   ├───__init__.py
│   ├───decoder.py
│   ├───dot_product_attention.py
│   ├───encoder.py
│   ├───model_config.py
│   ├───multi_head_attention.py
│   ├───position_embedding.py
│   ├───transformer_model.py
│   └───utils.py
├───training
│   ├───__init__.py
│   ├───data_converter.py
│   ├───inference.py
│   ├───model_builder.py
│   ├───prepare_data.py
│   └───train.py
├───README.md
├───requirements.txt
└───__init__.py
```

### Folder `data`
The `data`-folder contains the training data for this assignment.
It is also where the training script will write results like training log files, configuration and checkpoints.

The task in this assignment is english-to-german text translation.
The corresponding original training data in human-readable format can be found in `data/deu.txt`,
the data was downloaded from [this website](https://www.manythings.org/anki/) and originates from the [Tatoeba project](http://tatoeba.org/home).
The file contains more than 200.000 translated sentences of different complexity.

### Module `model`

The submodule `model` contains the transformer model implementation.
In this project, we will work with an encoder-decoder transformer model, which is implemented in `model/transformer_model.py`.
Its submodules and components are implemented as subclasses of `torch.nn.module` in the files
`position_embedding.py`, `dot_product_attention.py`, `multi_head_attention.py`, `encoer.py`, and `decoder.py`.

The file `model_config.py` contains a data class to capsule the configuration of a transformer model.
It includes methods for writing and loading model configuration to a text file.
This will help you keep track of your training configurations and to restore trained models,
in case you decide to experiment with the model parameters and run several trainings.

The file `position_embedding.py` contains an implementation of the positional embeddings as used in transformer models.
This class computes both, token-based and positional embeddings at the same time and combines them via addition,
as introduced in the lecture.
Contrary to the original transformer architecture, we employ learnable positional embeddings,
i.e. the model can decide for itself what embeddings assign to the sequence's positional indices.

### Module `training`

This module contains not only scripts for training and inference of the transformer model,
but also data handling and pre-processing routines.

The module's `__init__.py` contains path definitions and utility functions for working with directories and paths.
Further, special tokens (start, end, pad) are defined here.

The file `data_converter` is used to extract the sentence pairs from the original `.txt`-file 
and store the `str`-lists as `en-ge-all.pkl`, which makes the data more readily accessible.
Due to the binary format, it can no longer be inspected manually.

All the data pre-processing, like tokenization and assembly of the vocabulary are implemented in `prepare_data.py`.
We use `torchtext`'s simple word-to-token tokenizer, augmented by a few additional character replacements.
Naturally, the samples of german and english text vary in sequence length.
To be able for the transformer model to process all included sequences,
all input and output sequences must first be padded to the length of the maximum input and output sequence, respectively.
Therefore, a special padding token `<pad>` is used.
When using a padding token, we need to make sure our model and training procedures handle this token adequately.
In our case this includes
* Masking pad tokens for attention mechanisms (see padding masks in `model/utils.py`)
* Ignoring pad tokens in loss (and metrics) computation (via `ignore_index` parameter in loss and perplexity initialization)

The file also contains functions to store and load your preprocessed training data.
This is especially helpful, as preprocessing takes some time, and you do not want to perform it before each training anew.
When training the model using `train.py`'s function `train()`,
you can use its parameter `preprocess_data` to indicate whether preprocessing has to be performed (anew),
or if the preprocessed data should be loaded from disk (`data/dataset_preprocessed.pkl`).
You should run the preprocessing only once, before your first training.

The file `training/model_builder` contains functions to initialize the transformer model based on a given configuration,
and to load the stored weights of an already trained model to a `TransformerModel`.
(Note that the configuration of the model's stored weights and the initialized model need to be identical.)

The file `training/train.py` and `training/inference.py` contain scripts for training and inference of 
the transformer model for text translation (see tasks below).

### Directory `checks_and_tests`

This directory contains some useful tests to check your implementations.
These tests can conveniently be executed using the package `pytest`.
If you are using an IDE (e.g., PyCharm or VSCode), you should be able to run the tests via the context menu,
which opens on right-click on one of the `test_*.py`-files.
Otherwise, the tests can be invoked via command line, see [documentation](https://docs.pytest.org/en/stable/how-to/usage.html).

The provided tests a rather simple decency checks, which, for instance, confirm the dimensions of your returned tensors,
and ensure the code is runnable in general. 
In some cases, logical checks are performed.

Please note, that the provided tests are no proper unit tests as frequently used in software development.
(For instance, if you encounter a failure in the transformer's tests, 
this may also be caused by errors in dependencies (submodules) of the `TransformerModel`.) 

## Setup

Before you start working on the implementation tasks of this assignment, please make sure all required dependencies are installed to your Python environment.
To do so, please run the command `pip install -r requirements.txt` in your command line (current working directory should be this repository's base path).

Next, please open the file `training/train.py` and fill your personal information in the line comments at the very top of the file.

Run the script `training/data_converter.py` to create the file `en-ge-all.pkl`.

## Tasks

### 1) Implement Dot-Product Attention

Your first task is to finish the implementation of a simple (masked) dot-product attention mechanism.

The module's forward pass receives query, key and value matrices, and, optionally, a corresponding mask matrix, in case masked attention score is computed.
The computation and masking of the scores is already implemented.
What is left to do, is to compute the value weights according to the computed scores
and, subsequently, the weighted sum of the given values as output of the forward pass.
Only the resulting matrix (tensor) of weighted values is returned.

Once you are done with your implementation, run the tests in file `checks_and_tests/dot_product_attention.py`.
They perform simple sanity checks of your implementation.
We advise you to move on to the next task only after the tests have been passed.

If you are using an IDE like VSCode or PyCharm, you should be able to run the tests within the test source files via 
the context menus which open on right-click onto the files containing the tests.

**File**: `model/dot_product_attention.py`

**Tests**: `test_dot_product_attention.py`

### 2) Implement Multihead Attention

Next, we implement a module for the transformer's multihead attention.
This includes the projection of input key, query and values to their respective dimension.
The vectors are split into segments according to the number of heads which are computed.
The key, query and value vector for the individual heads are stacked in a single tensor each,
such that the previously implemented dot-product attention class can be used to compute the attention
in all heads simultaneously.
Once the attention output weighted values are computed, the vectors are once again concatenated 
to a single vector containing the resulting values for all the attention heads.
Last, a linear layer which projects the attention results vector back to the transformer model's internal dimension is applied within the multihead attention module.  

The given implementation already contains a method to handle the cutting and stacking of multiple heads before attention computation (`_expand_heads_and_reorder`)
and another method for the re-concatenation of the stacked heads to the original size (`_concat_heads_and_reorder`).

**File**: `model/multi_head_attention.py`

**Tests**: `test_multihead_attention.py`

### 3) Implement Encoder Layer

Note first, that the file `model/encoder.py` contains two classes:
* The class ``EncoderLayer`` represents a single layer of the encoder
* The class ``Encoder`` represents the model's complete encoder consisting of one or multiple layers

The class `Encoder` is already fully implemented.
In its `forward()` method, tt computes the input's embeddings and runs them through all its layers.

The class `EncoderLayer` is where the actual encoder magic happens.
Your task is to complete the implementation of its `forward()`-method.
As seen in the lecture, the forward pass consists of the following steps:
* Apply self-attention
* Subsequent add-and-norm layer
* Feed-forward layers
* Again followed by an add-and-norm layer

Dropout is used in between layers as regularization.
Refer to the lecture and the comments in the code for more details.

**File**: `model/encoder.py`

**Tests**: `test_encoder.py`

### 4) Implement Decoder Layer

Consistent to the encoder's file, the file `model/decoder.py`, contains two classes.
Similarly, the single decoder layer needs to be completed as part of this assignment,
the consolidation of multiple decoder layers into the decoder module is already implemented.

**File**: `model/decoder.py`

**Tests**: `test_decoder.py`

### 5) Implement Transformer Model

After encoder and decoder are complete, the complete transformer model can be assembled in the `model/transformer_model.py`.
Implement the transformer forward pass following the instruction in the comments.

After the implementation of the transformer is completed,
all tests in directory `checks_and_tests` should pass without any errors.

**File**: `model/transformer_model.py`

**Tests**: `test_transformer.py`

### 6) Start a Training

Once the above tasks are completed and all tests are passed, you can start training the transformer model.
Note, that there are two model configurations already implemented in `train.py`:
a smaller and a larger model.
(You can select which model to train by setting `model_config = model_config_small` or `model_config = model_config_large` in `train.py`.)
Both models require a GPU to run the complete training to convergence.
For this assignment, it is sufficient to run either (or your own) configuration for at least one training epoch.

If you start training, a new directory named `training_*` will be created in your `data`-directory.
The model configuration, the training log files and the weight checkpoint (after each complete epoch)
will be written to this directory.

*Note:* When you run `train()` for the first time, data preprocessing has to be performed,
and the preprocessed data is stored to the file `data/dataset_preprocessed.pkl`.
After that, you can speed up your training by setting the `train()`-functions parameter `preprocess_data=False`.
In this case, data preprocessing will not be performed again, the already preprocessed data will be loaded instead.

### 7) Run the inference script

Once you have trained the model for at least one complete epoch,
and a model checkpoint file (`transformer_model_*.pkl`) has been written to the training directory,
you can run the inference script.
The script should automatically load your newest checkpoint and run a few example sentences.
Results are printed to the console.

Just a small disclaimer: We do not expect mind-blowing translation results for several reasons.
Namely, the training data set is relatively small,
our model implementations are kept simple for educational purposes,
and we are working with very limited hardware resources (as compared to state-of-the-art data sets and setups).


### Assignment Requirements

To pass the bonus assignment the following criteria should be met:
* Solve the implementation tasks above
* All tests in `checks_and_tests` succeed
* Train a model for at least one epoch and provide training folder (containing log file, config file, checkpoint)
* The inference script can be run employing your model to obtain translations

(If you run multiple trainings, please do not include all the training folders in your submission.)

In case of any problems or issues, please don't hesitate to [contact](mailto:laura.doerr@kit.edu) us.
We are happy to help!