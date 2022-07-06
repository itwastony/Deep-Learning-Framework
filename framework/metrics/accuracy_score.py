import numpy as np


def accuracy_score(y_pred, y_true):
    size = y_pred.shape[0]
    true_label = y_true.argmax(axis=1)
    pred_label = y_pred.argmax(axis=1)

    correct = (true_label == pred_label).sum()

    return correct / size
