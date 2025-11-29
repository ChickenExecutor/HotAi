import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader

from config import model_best_checkpoint
from data_loader import get_train_test_data
from model import MyModel
from train import predict


def evaluate():
    data_train, data_val, data_test, mean, std = get_train_test_data()

    dl_test = DataLoader(data_val, batch_size=1024, shuffle=False, num_workers=1, pin_memory=True)

    model = MyModel()

    model_data = torch.load(model_best_checkpoint)
    model.load_state_dict(model_data["model_state"])

    true_classes, predicted_classes = predict(model, dl_test)

    accuracy = accuracy_score(true_classes, predicted_classes)
    print("Accuracy: {:.4f}".format(accuracy))

    cm_values = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(cm_values, display_labels=dl_test.dataset.dataset.classes)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    evaluate()
