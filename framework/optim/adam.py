import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, model, beta1=0.9, beta2=0.999):
        super(Adam, self).__init__()

    def step(self, lr=1e-3):
        pass
