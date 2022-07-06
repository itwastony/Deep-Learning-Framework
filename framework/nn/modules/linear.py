import numpy as np
from .module import Module


class Linear(Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(Linear, self).__init__()

        dispersion = 2 / (in_dim + out_dim)
        self.W = np.random.normal(
            loc=0, scale=np.sqrt(dispersion), size=(in_dim, out_dim))
        self.b = np.random.normal(
            loc=0, scale=np.sqrt(dispersion), size=(1, out_dim))

    def forward_pass(self, input_data: np.ndarray):
        self.output = np.dot(input_data, self.W) + self.b

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        self.grad_W = np.dot(input_data.T, grad_output)
        grad_input = np.dot(grad_output, self.W.T)

        return grad_input

    def parameters(self):
        return [self.W, self.b]

    def grad_parameters(self):
        return [self.grad_W, self.grad_b]
