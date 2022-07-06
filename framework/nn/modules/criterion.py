import numpy as np

class Criterion():
    def forward_pass(self, pred_data: np.ndarray, target_data: np.ndarray):
        raise NotImplementedError
    
    def backward_pass(self, pred_data: np.ndarray, target_data: np.ndarray):
        raise NotImplementedError