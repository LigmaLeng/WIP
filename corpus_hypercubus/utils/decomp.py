from . import xp
from .import numpyPack
from typing import Tuple
import numpy as np

class PCA:
    def __init__(self, X:np.ndarray|str, n_components:int, testing:bool=False) -> None:
        self.n_components = n_components
        self.X = np.load(X) if isinstance(X, str) else X
        self.n_samples, self.m_features = self.X.shape
        self.mu = self.X.mean(axis=0)
        self.X -= self.mu
        self.total_var = np.var(self.X, axis=0, ddof=1).sum()
        self.testing = testing
        if testing:
            self.randargs = np.random.choice(self.n_samples, size=self.n_samples//50, replace=False)
            self.test_copies = self.X[self.randargs].copy()

    def compute(self, discount_sampling:int=1, oversamples:int=10, power_iter:int=20) -> None:
        self.U, self.S, self.VT = SVD_EconomyPlus(self.X, self.n_components, discount_sampling=discount_sampling, oversamples=oversamples, power_iter=power_iter)
        self.compile_stats()
        self.check_test()

    def compile_stats(self):
        self.explained_variance = self.S**2 / (self.n_samples-1)
        self.explained_variance_ratio = self.explained_variance / self.total_var
        if self.m_features > self.n_components:
            self.noise_variance = (self.total_var - self.explained_variance.sum()) / self.m_features - self.n_components
        else: self.noise_variance = np.asarray([0.0], dtype=np.float32)

    def save(self, fname:str):
        np.savez(f"{fname}",
                 mean=self.mu, U=self.U, S=self.S, VT=self.VT,
                 explained_variance=self.explained_variance,
                 explained_variance_ratio=self.explained_variance_ratio,
                 noise_variance=self.noise_variance
                 )

    def check_test(self):
        if not self.testing:
            return
        assert self.n_components >= self.m_features, "Decomposition accuracy testing only implemented for equal number of features"
        test_componenets = ((self.U * self.S) @ self.VT)[self.randargs]
        try:
            np.testing.assert_array_almost_equal(self.test_copies, test_componenets, decimal=5)
        except:
            print("pca precision less than 5 decimal places")

def SVD_EconomyPlus(arr, n_components, discount_sampling:int=1, oversamples:int=10, power_iter:int=None):
    n_samples, m_features = arr.shape
    subcomponents = n_components // discount_sampling
    U = np.empty(shape=(n_samples, n_components), dtype=np.float32)
    S = np.empty(shape=(n_components), dtype=np.float32)
    VT = np.empty(shape=(n_components, m_features), dtype=np.float32)
    for i in range(discount_sampling):
        if i > 0 :
            _U = U[:, : i*subcomponents]
            _S = S[: i*subcomponents]
            _VT = VT[: i*subcomponents]
            X = xp.asarray(arr - (_U * _S) @ _VT, dtype=xp.float32)
        else:
            X = xp.asarray(arr, dtype=xp.float32)

        _range = range(i*subcomponents, (i+1)*subcomponents)
        print(f"component set {i+1}")
        U[:, _range], S[_range], VT[_range] = randomisedSVD(X, subcomponents, oversamples=oversamples, power_iter=power_iter)

    return U, S, VT

def randomisedSVD(arr:np.ndarray|xp.ndarray, n_components:int, oversamples:int=10, power_iter:int=None, flip=True) -> Tuple[np.ndarray|xp.ndarray,...]:
    isNumpy = isinstance(arr, np.ndarray)

    n_Q_vectors = n_components + oversamples
    n, m = arr.shape

    if not power_iter:
        power_iter = 7 if n_components < min(n,m)//10 else 4
    
    Q = mimic_orthonormal(arr, n_Q_vectors, power_iter)
    B = Q.T @ arr

    Uhat, S, VT = xp.linalg.svd(B, full_matrices=0)
    del B
    
    U = Q @ Uhat
    if flip:
        signs = xp.sign(U[xp.argmax(xp.abs(U), axis=0), range(U.shape[1])])
        U *= signs
        VT *= signs[:, xp.newaxis]

    U = U[:, :n_components]
    S = S[:n_components]
    VT = VT[:n_components]

    if isNumpy:
        return U, S, VT
    else:
        return numpyPack(U, S, VT)

def mimic_orthonormal(arr:xp.ndarray, size:int, power_iter:int) -> xp.ndarray:
    Q = xp.random.normal(size=(arr.shape[1], size), dtype=xp.float32)
    for i in range(power_iter):
        print(f"pwr_iter: {i+1:d}", end="\r")
        Q, _ = xp.linalg.qr(arr @ Q)
        Q, _ = xp.linalg.qr(arr.T @ Q)

    Q, _ = xp.linalg.qr(arr @ Q)
    return Q

def eigen_decomp(self, X:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covar =  (X.T @ X) / (X.shape[0]-1)
    eigval, eigvec = np.linalg.eig(covar)
    args = np.argsort(np.abs(eigval))[::-1]
    eigval = eigval[args]
    eigvec = eigvec[:, args]
    return eigval, eigvec


def pin_to_memory(arr):
    mem = xp.cuda.alloc_pinned_memory(arr.nbytes)
    pinned = np.frombuffer(mem, arr.dtype, arr.size).reshape(arr.shape)
    pinned[...] = arr
    return pinned