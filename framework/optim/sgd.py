import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, model):
        super(SGD, self).__init__()

        self.model = model

    def step(self, lr=1e-3):
        for weights, gradients in zip(self.model.parameters(), self.model.grad_parameters()):
            weights -= lr * gradients
