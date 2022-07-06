import numpy as np
from .module import Module


class Sequential(Module):
    """Sequential will wrap list of modules and it's consistently will running them.
    """

    def __init__(self, *layers: list):
        super(Sequential, self).__init__()

        self.layers = layers

    def forward_pass(self, input_data: np.ndarray):
        for layer in self.layers:
            input_data = layer.forward_pass(input_data)

        self.output_data = input_data
        return self.output_data

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        for i in range(len(self.layers)-1, 0, -1):
            grad_output = self.layers[i].backward_pass(
                self.layers[i-1].output, grad_output)

        grad_input = self.layers[0].backward_pass(input_data, grad_output)

        return grad_input

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()

        return params

    def grad_parameters(self):
        grad_params = []
        for layer in self.layers:
            grad_params += layer.grad_parameters()

        return grad_params

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()
