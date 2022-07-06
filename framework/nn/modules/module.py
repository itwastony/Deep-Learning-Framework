import numpy as np


class Module():
    """ The abstract class from which the layers of our neural network will inherit 
    """

    def __init__(self):
        self._train_mode = True

    def forward_pass(self, input_data: np.ndarray):
        raise NotImplementedError

    def backward_pass(self, input_data: np.ndarray, grad_output: np.ndarray):
        raise NotImplementedError

    def parameters(self):
        "it's returning own parameters"
        return []

    def grad_parameters(self):
        "it's returning own gradient parameters"
        return []

    def train(self):
        self._train_mode = True

    def eval(self):
        self._train_mode = False
