import numpy as np
from .criterion import Criterion
from .. import functional as F


class MSELoss(Criterion):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward_pass(self, pred_data, target_data):
        self.output = F.mse_loss(pred_data, target_data)

        return self.output

    def backward_pass(self, pred_data, target_data):
        batch_size = pred_data.shape[0]
        self.grad_output = 2 * (pred_data - target_data) / batch_size

        return self.grad_output


class CrossEntropyLoss(Criterion):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward_pass(self, pred_data, target_data):
        self.output = F.crossentropy_func(pred_data, target_data)

        return self.output

    def backward_pass(self, pred_data, target_data, eps=1e-10):
        pred_data = np.clip(pred_data, eps, 1. - eps)
        self.grad_output = F.softmax_forward(pred_data) - target_data

        return self.grad_output
