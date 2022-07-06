import numpy as np


def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


def softmax_forward(x):
    max_element = np.amax(x, axis=-1, keepdims=True)
    e = np.exp(x - max_element)
    softmax_result = e / np.sum(e, axis=-1, keepdims=True)

    return softmax_result


# def softmax_backward(x, flag=True):
#     for row_idx in range(x.shape[0]):
#         s = softmax_forward(x[row_idx]).reshape(-1, 1)    
#         jac = np.diagflat(s) - np.dot(s, s.T)

#         grad_obj = np.dot(jac, x[row_idx])
#         grad_obj = grad_obj.reshape(1, -1)

#         if flag:
#             grad_input_x = grad_obj
#             flag = False

#         else:
#             grad_input_x = np.vstack((grad_input_x, grad_obj))

#     return grad_input_x


def tanh_func(x):
    return 2 * sigmoid_func(2 * x) - 1


def mse_loss(y_pred, y_target):
    batch_size = y_pred.shape[0]

    return np.sum(np.power(y_pred - y_target, 2)) / batch_size


def crossentropy_func(y_pred, y_target, eps=1e-10):
    num_samples = y_pred.shape[0]
    y_pred = np.clip(y_pred, eps, 1. - eps)
    ce = -np.sum(y_target * np.log(y_pred))

    return ce / num_samples


def batchnorm_forward(X, gamma, beta, moving_mean,
                      moving_var, eps, momentum, train_mode):
    if train_mode:
        mean = X.mean(axis=0)
        var = np.power(X - mean, 2).mean(axis=0)
        X_hat = (X - mean) / np.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var

    else:
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)

    Y = gamma * X_hat + beta

    return Y, moving_mean, moving_var


def batchnorm_backward(X, gamma, beta, eps, grad_out):
    N, D = X.shape

    mu = 1./N * np.sum(X, axis=0)
    xmu = X - mu
    sq = xmu ** 2
    var = 1./N * np.sum(sq, axis=0)
    sqrtvar = np.sqrt(var + eps)
    ivar = 1./sqrtvar
    xhat = xmu * ivar

    dbeta = np.sum(grad_out, axis=0)
    dgamma = np.sum(grad_out * xhat, axis=0)
    dxhat = grad_out * gamma
    divar = np.sum(dxhat * xmu, axis=0)
    dxmu1 = dxhat * ivar
    dsqrtvar = -1. / (sqrtvar**2) * divar
    dvar = 0.5 * 1. / np.sqrt(var+eps) * dsqrtvar
    dsq = 1. / N * np.ones((N, D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
    dx2 = 1. / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2

    return dx, dgamma, dbeta
