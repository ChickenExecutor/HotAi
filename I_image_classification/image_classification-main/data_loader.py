import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from tqdm import tqdm

from config import data_dir

URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"


def random_split(dataset, val_ratio=0.2, test_ratio=0.2, random_state=32):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n_val = int(len(dataset) * val_ratio)
    n_test = int(len(dataset) * test_ratio)
    n_train = len(dataset) - n_val - n_test
    split = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
    return split


def download_data(target_dir, subdir_name):
    if data_dir.startswith("~/work/__shared/"):
        return  # assume data is already downloaded in bwJupyter environment
    if not check_integrity(os.path.join(target_dir, "EuroSAT.zip")):
        download_and_extract_archive(URL, target_dir, md5=MD5)
        os.rename(os.path.join(target_dir, "2750"), os.path.join(target_dir, subdir_name))


def calc_normalization(train_dl: torch.utils.data.DataLoader):
    # Calculate the mean and std of each channel on images from `train_dl`
    mean = torch.zeros(3)
    mean_squared = torch.zeros(3)
    n = len(train_dl)
    for images, labels in tqdm(train_dl, "Compute normalization"):
        mean += images.mean([0, 2, 3]) / n
        mean_squared += (images ** 2).mean([0, 2, 3]) / n
    var = mean_squared - mean ** 2
    return mean, var.sqrt()


def get_train_test_data():
    download_data(data_dir, "EuroSAT")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor()
        ]
    )

    image_folder = ImageFolder(os.path.join(data_dir, "EuroSAT"), transform)
    train_data, val_data, test_data = random_split(image_folder, 0.2, 0.2, 32)

    mean, std = calc_normalization(DataLoader(train_data))

    # mean = torch.Tensor([0.3449, 0.3804, 0.4080])
    # std = torch.Tensor([0.2033, 0.1374, 0.1159])

    image_folder.transform.transforms.append(transforms.Normalize(mean, std))

    return train_data, val_data, test_data, mean, std


if __name__ == '__main__':
    get_train_test_data()
