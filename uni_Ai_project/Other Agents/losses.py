import cupy as np

def mse(v, v_hat):
    """
    mean squared error is modified in typical implementations 
    where E = 1/2N * SUM(V - V_hat)^2
    to get a derivative of the Error that is smoother to compute
    without any noticable sacrifices:
        dE_dV = 1/N (V_hat - V)
    """
    return np.mean(np.power(v - v_hat, 2)) / 2

def mse_prime(v, v_hat):
    return (v_hat - v) / v.size

def log_loss(v, v_hat):
    return (- np.sum(v * np.log(v_hat) + (1 - v) * np.log(1 - v_hat)))

def log_loss_prime(v, v_hat):
    return (-(v - v_hat) - (1 - v) / (1 - v_hat))

def cross_entropy(v, v_hat, epsilon=1e-12):
    target = np.argmax(v, axis=0)
    clipped = np.clip(v_hat[target, 0], epsilon, 1. - epsilon)
    return -np.log(clipped)

def cse_softmax_prime(v, v_hat):
    target = np.argmax(v, axis=0)
    v_hat[target] -= 1
    return v_hat

