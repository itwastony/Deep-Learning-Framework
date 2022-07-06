import numpy as np
from .module import Module
from .. import functional as F


class BatchNorm(Module):
    def __init__(self, num_features, epsilon=1e-4, momentum=0.9):
        super(BatchNorm, self).__init__()

        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.moving_mean = np.zeros((1, num_features))
        self.moving_var = np.ones((1, num_features))
        self.eps = epsilon
        self.momentum = momentum

    def forward_pass(self, input_data: np.ndarray):
        self.output, self.moving_mean, self.moving_var = F.batchnorm_forward(
            input_data, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=self.eps,
            momentum=self.momentum, train_mode=self._train_mode)

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        grad_input, self.grad_gamma, self.grad_beta = F.batchnorm_backward(
            input_data, self.gamma, self.beta, self.eps, grad_output)

        return grad_input

    def parameters(self):
        return [self.gamma, self.beta]

    def grad_parameters(self):
        return [self.grad_gamma, self.grad_beta]
