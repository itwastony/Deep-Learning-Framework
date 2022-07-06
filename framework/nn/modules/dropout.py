import numpy as np
from .module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def forward_pass(self, input_data: np.ndarray):
        if self._train_mode:
            self.mask = np.random.binomial(
                n=1, p=self.p, size=input_data.shape) / self.p
            self.output = input_data * self.mask

        else:
            self.output = input_data

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        if self._train_mode:
            grad_input = grad_output * self.mask

        else:
            grad_input = grad_output

        return grad_input
