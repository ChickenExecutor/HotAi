import datetime
import shutil
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from data_loader import get_train_test_data
from model import MyModel
from config import current_dir, model_checkpoint_file, model_best_checkpoint

### STUDENT NAME: Keanu Seiraffi
### STUDENT IDENTIFIER (U****): ubhqg
### COLLABORATION WITH:

# if you are using a device with one or multiple GPU(s), employment of such may speed up computations
# e.g. change the following line to DEVICE = "cuda:0"
# you can check device availability with the following lines of code:
### torch.cuda.device_count()
# if a value greater than 0 is returned, a GPU is available for use with pytorch
DEVICE = "cuda"

# TO DO: find appropriate training hyperparameters
# YOUR CODE HERE
epochs = 100
learning_rate = 0.001
batch_size = 100

class State:
    best_acc = 0
    writer: SummaryWriter = None
    normalization = None


@torch.no_grad()
def predict(model: nn.Module, dl: torch.utils.data.DataLoader, show_progress=True):
    if show_progress:
        dl = tqdm(dl, "Predict", unit="batch")
    device = next(model.parameters()).device

    model.eval()
    preds = []
    truth = []
    for images, labels in dl:
        images = images.to(device, non_blocking=True)
        truth += labels.tolist()

        # TO DO: pass images through model, reformat predictions and write them to the preds list:
        # The preds is a list of all class predictions for the processed images. For each image passed through the model, it contains a single integer indicating the predicted class.
        # YOUR CODE HERE
        outputs = model(images)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        batch_preds = outputs.argmax(dim=1)
        preds += batch_preds.cpu().tolist()

    return torch.as_tensor(truth), torch.as_tensor(preds)


def train_epoch(train_dl, model, loss, optimizer, epoch):
    model.train()
    train_dl = tqdm(train_dl, "Train", unit="batch")
    for i, (images, labels) in enumerate(train_dl):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        batch_predictions = model(images)
        batch_loss = loss(batch_predictions, labels)
        batch_accuracy = (labels == batch_predictions.argmax(1)).float().mean()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        State.writer.add_scalar('loss/train', batch_loss, epoch * len(train_dl) + i)
        State.writer.add_scalar('acc/train', batch_accuracy, epoch * len(train_dl) + i)


def train():
    data_train, data_val, data_test, mean, std = get_train_test_data()

    dl_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    dl_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    model = MyModel()
    model = model.to(DEVICE)
    loss = nn.CrossEntropyLoss()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)

    # initialize summary writer. logs are written to ./runs/[DATETIME]
    State.writer = SummaryWriter(os.path.join(current_dir,
                                              f"runs/{datetime.datetime.now().strftime("%Y%m%d_%H.%M.%S")}"))
    # display some examples in tensorboard
    images, labels = next(iter(dl_train))
    originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    State.writer.add_images('images/original', originals, 0)
    State.writer.add_images('images/normalized', images, 0)

    for epoch in trange(epochs, desc="Epochs"):
        train_epoch(dl_train, model, loss, optimizer, epoch)
        truth, preds = predict(model, dl_val)

        torch.save(
            {'normalization': State.normalization, 'model_state': model.state_dict()},
            model_checkpoint_file,
        )

        val_acc = (truth == preds).float().mean()
        State.writer.add_scalar('acc/val', val_acc, epoch * len(dl_train))
        if val_acc > State.best_acc:
            print(f"New best validation accuracy: {val_acc}")
            State.best_acc = val_acc
            shutil.copy(model_checkpoint_file, model_best_checkpoint)


if __name__ == '__main__':
    train()
