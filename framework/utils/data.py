import numpy as np


def data_loader(X, Y, batch_size=1):
    num_objects = X.shape[0]

    indices = np.arange(num_objects)
    np.random.shuffle(indices)

    for start in range(0, num_objects, batch_size):
        end = min(start + batch_size, num_objects)
        batch_idx = indices[start:end]

        yield X[batch_idx], Y[batch_idx]
