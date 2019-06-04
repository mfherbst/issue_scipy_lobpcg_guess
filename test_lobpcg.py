#!/usr/bin/env python3
import numpy as np

from numpy.random import randn

import scipy

from scipy.linalg import eigh, eigvalsh, orth
from scipy.sparse.linalg import lobpcg

A = np.load("./data.npz")["A"]
P = np.load("./data.npz")["P"]
x0 = np.load("./data.npz")["x0"]

N, m = x0.shape
λreff, vreff = eigh(A)
λref = λreff[:m]
vref = vreff[:, :m]

print("scipy verison: ", scipy.__version__)
print("Problem properties:")
print("N:                 ", N)
print("m:                 ", m)
print("λref:              ", λref)
print("λapprox:           ", eigvalsh(x0.T @ A @ x0))
print("A λ dist:          ", np.abs(λreff[m+1] - λreff[m]))

λPAm = vreff[m].T @ P @ A @ vreff[m]
λPAm1 = vreff[m + 1].T @ P @ A @ vreff[m + 1]
print("P*A ritz dist:     ", np.abs(λPAm1 - λPAm))


def test_lobpcg(read_guess=True, tol=1e-5):
    X = orth(randn(N, m))
    if read_guess:
        X = x0

    #try:
        λ, v = lobpcg(A, X, largest=False, M=P, maxiter=100, tol=tol)
        assert np.max(np.abs(λref - λ)) < tol
    #except np.linalg.LinAlgError as e:
    #    print("ERROR: ", str(e))


print()
print("read_guess=True")
test_lobpcg(read_guess=True)

print()
print("read_guess=False")
test_lobpcg(read_guess=False)
