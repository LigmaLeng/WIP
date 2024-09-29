from . import xp
__all__ = ["MeanSquaredError", "BinaryCrossEntropy", "COR", "Softmax_Objective"]

class MeanSquaredError:
    @staticmethod
    def loss(v, v_hat):
        """
        Technically not the mean squared error but its just halving 
        the error so we wont have to consider the exponent term in backpropagation
        ultimately it just scales the backpropagation and will be adjusted for via learning rate etc.
        where E = 1/2N * (V - V_hat)**2
        to get a derivative of the Error :
            dE_dV = 1/N (V - V_hat) * -1
                    1/N (V_hat - V)
        """
        return xp.mean((v - v_hat)**2) / 2
    @staticmethod
    def grad(v, v_hat):
        return (v_hat - v) / len(v)

class BinaryCrossEntropy:
    @staticmethod
    def loss(v, v_hat):
        # return - xp.sum(v * xp.log(v_hat) + (1 - v) * xp.log(1 - v_hat))
        return - xp.sum(v * xp.log(v_hat) + (1 - v) * xp.log(1 - v_hat))
    @staticmethod
    def grad(v, v_hat):
        return -(v / v_hat - (1 - v) / (1 - v_hat))

class COR:
    @staticmethod
    def logsigmoid(x):
        return - xp.log((1 + xp.exp(-x)))

    @staticmethod
    def sigmoid(x):
        return 1/(1 + xp.exp(-x))

    @staticmethod
    def d_log_sigmoid(x):
        return 1/(1 + xp.exp(x))

    def loss(v, v_hat):
        probas = COR.sigmoid(v_hat)
        loss = 0
        total_samples = 0
        for i in range(v_hat.shape[-1]):
            train_subsets = v > i-1
            label_targets = (v[train_subsets] > i).astype(xp.float32)
            if len(label_targets) < 1:
                continue
            total_samples += len(label_targets)
            preds = v_hat[train_subsets, i]
            loss += - xp.sum(COR.logsigmoid(preds) * label_targets
                            + (COR.logsigmoid(preds) - preds) * (1 - label_targets))
        return loss/total_samples, probas

    def grad(v, v_hat):
        grads = xp.zeros_like(v_hat)
        total_samples = 0
        for i in range(v_hat.shape[-1]):
            train_subsets = v > i-1
            label_targets = (v[train_subsets] > i).astype(xp.float32)
            if len(label_targets) < 1:
                continue
            total_samples += len(label_targets)
            preds = v_hat[train_subsets, i]
            grads[train_subsets, i] += - (COR.d_log_sigmoid(preds) * label_targets
                            + (COR.d_log_sigmoid(preds) - 1) * (1 - label_targets))
        return grads / total_samples


class Softmax_Objective:
    @staticmethod
    def loss(v, v_hat, epsilon=1e-12):
        targets = xp.argmax(v, axis=1, keepdims=True)
        return (-xp.log(xp.clip(xp.take_along_axis(v_hat, targets, axis=1), epsilon, 1. - epsilon))).sum() / len(v_hat)
    
    @staticmethod
    def grad(v, v_hat):
        target = xp.argmax(v, axis=1)
        v_hat[xp.arange(v_hat.shape[0]), target] -= 1
        return v_hat / len(v_hat)
