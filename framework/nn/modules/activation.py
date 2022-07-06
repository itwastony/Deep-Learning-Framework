import numpy as np
from .module import Module
from .. import functional as F


class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward_pass(self, input_data: np.ndarray):
        self.output = np.maximum(input_data, 0)

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        grad_input = np.multiply(grad_output, input_data > 0)

        return grad_input


class LeakyReLU(Module):

    def __init__(self, slope=0.02):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def forward_pass(self, input_data: np.ndarray):
        self.output = np.where(input_data > 0, input_data,
                               self.slope * input_data)

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        grad_input = np.where(input_data > 0, grad_output,
                              self.slope * grad_output)

        return grad_input


class Sigmoid(Module):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward_pass(self, input_data: np.ndarray):
        self.output = F.sigmoid_func(input_data)

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        grad_input = grad_output * \
            (F.sigmoid_func(input_data) * (1 - F.sigmoid_func(input_data)))

        return grad_input


class Tanh(Module):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward_pass(self, input_data: np.ndarray):
        self.output = F.tanh_func(input_data)

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        tanh_grad = 1 - np.power(F.tanh_func(input_data), 2)
        grad_input = tanh_grad * grad_output

        return grad_input


class SoftMax(Module):

    def __init__(self):
        super(SoftMax, self).__init__()

    def forward_pass(self, input_data: np.ndarray):
        self.output = F.softmax_forward(input_data)

        return self.output

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        # grad_input = F.softmax_backward(input_data) * grad_output

        return grad_output
