import datetime
import os

if os.getenv("HOTAI_TRANSFORMER_ENV", "UNDEFINED").lower() == "bwjupyter":
    data_read_only_dir = "/home/jovyan/work/__shared/data"
else:
    data_read_only_dir = os.path.join(os.path.dirname(__file__), "../data")

data_dir = os.path.join(os.path.dirname(__file__), "../data")

data_file_write_path = os.path.join(data_dir, "en-ge-all.csv")
preprocessed_data_file_write_path = os.path.join(data_dir, "dataset_preprocessed.pkl")

training_dir_format = os.path.join(data_dir, "training_{}")
training_dir = training_dir_format.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

config_file_name = "model_config.json"
model_state_dict_file_name = "transformer_model_{}.pkl"

start_token = "<start>"
eos_token = "<end>"
pad_token = "<>"


def get_training_dirs():
    training_dirs = []
    for entry in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, entry)):
            if entry.startswith("training"):
                dir = os.path.join(data_dir, entry)
                training_dirs.append(dir)
    return training_dirs


def get_model_state_dict_files(training_dir):
    model_files = {}
    for entry in os.listdir(training_dir):
        if os.path.isfile(os.path.join(training_dir, entry)):
            if entry.startswith("transformer_model"):
                epoch_number = int(entry.rsplit(".", 1)[0].rsplit("_", 1)[1])
                model_files[epoch_number] = os.path.join(training_dir, entry)
    return model_files

def get_existing_preprocessed_data_file():
    if os.path.isfile(os.path.join(data_dir, "dataset_preprocessed.pkl")):
        return os.path.join(data_dir, "dataset_preprocessed.pkl")
    elif os.path.isfile(os.path.join(data_read_only_dir, "dataset_preprocessed.pkl")):
        return os.path.join(data_read_only_dir, "dataset_preprocessed.pkl")
    else:
        return None

def get_existing_data_file_path():
    if os.path.isfile(os.path.join(data_dir, "en-ge-all.csv")):
        return os.path.join(data_dir, "en-ge-all.csv")
    elif os.path.isfile(os.path.join(data_read_only_dir, "en-ge-all.csv")):
        return os.path.join(data_read_only_dir, "en-ge-all.csv")
    else:
        return None