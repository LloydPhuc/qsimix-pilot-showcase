"""Small NumPy A0 statevector decoder, batched over pixels and bands.

Canonical input A[E,N], M[B,E]; outputs [B,N]. No shots or learned readout.
The independent scalar implementation stays in validation/ for regression QA.
"""
from __future__ import annotations

import numpy as np


class ResidualModel:
    def __init__(self, M, theta0, theta=None, lam=0.03):
        self.M = np.array(M, dtype=np.float64, copy=True)
        self.theta0 = np.array(theta0, dtype=np.float64, copy=True)
        self.theta = self.theta0.copy() if theta is None else np.array(theta, dtype=np.float64, copy=True)
        if (self.M.ndim != 2 or self.M.shape[1] != 4
                or self.theta0.shape != (14,) or self.theta.shape != (14,)
                or not all(np.isfinite(x).all() for x in (self.M, self.theta0, self.theta))
                or not np.isfinite(lam) or lam <= 0):
            raise ValueError("Pilot requires finite M[B,4], 14 parameters and positive lambda")
        self.lam = float(lam)
        self.theta0.setflags(write=False)
        states = np.arange(16)
        self.z = np.array([1 - 2 * ((states >> (3 - i)) & 1) for i in range(4)])
        self.obs = self.z.mean(axis=0)
        self.pairs = [(np.flatnonzero(self.z[i] == 1), np.flatnonzero(self.z[i] == -1))
                      for i in range(4)]
        self.phase = [np.exp(-0.5j * np.pi * self.M[:, i, None] * self.z[i]) for i in range(4)]
        self.reference_vertices = self.q(np.eye(4), self.theta0)
        self.vertices = self.q(np.eye(4), self.theta)

    def q(self, A, theta):
        a = np.asarray(A, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != 4 or not np.isfinite(a).all():
            raise ValueError("Expected finite A[4,N]")
        v = np.zeros((a.shape[1], self.M.shape[0], 16), dtype=np.complex128)
        v[..., 0] = 1

        def rotate(wire, angle):
            lo, hi = self.pairs[wire]
            x, y = v[..., lo].copy(), v[..., hi].copy()
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            v[..., lo], v[..., hi] = c * x - s * y, s * x + c * y

        for pars in np.asarray(theta).reshape(2, 7):
            for i in range(4):
                rotate(i, np.pi * a[i, :, None, None])
                v *= self.phase[i][None, :, :]
            for i in range(3):
                v *= np.exp(-0.5j * pars[4 + i] * self.z[i] * self.z[i + 1])
            for i in range(4):
                rotate(i, pars[i])
        return (np.abs(v) ** 2 @ self.obs).T

    def anchored(self, A, theta):
        return self.q(A, theta) - self.q(np.eye(4), theta) @ A

    def forward(self, A):
        # Reference is re-evaluated for the changing abundance, even when frozen.
        delta = self.q(A, self.theta) - self.q(A, self.theta0)
        return self.M @ A + self.lam * (delta - (self.vertices - self.reference_vertices) @ A)

    def jacobian(self, a, step=1e-6):
        """Central differences of the complete decoder, including reference/anchors."""
        directions = step * np.eye(4)
        return (self.forward(a[:, None] + directions)
                - self.forward(a[:, None] - directions)) / (2 * step)


class PairwiseModel:
    """Six shared, signed quadratic coefficients; anchored classical control.

    F_b(a) = M_b a + sum_{i<j} beta_ij M_bi M_bj a_i a_j.
    Signed unconstrained beta means this is NOT a physical GBM fit.
    """
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    def __init__(self, M, coefficients):
        self.M = np.array(M, dtype=np.float64, copy=True)
        self.coefficients = np.array(coefficients, dtype=np.float64, copy=True)

    def features(self, A):
        return np.stack([self.M[:, i, None] * self.M[:, j, None] * A[i] * A[j]
                         for i, j in self.pairs], axis=-1)

    def forward(self, A):
        return self.M @ A + self.features(A) @ self.coefficients

    def jacobian(self, a, step=1e-6):
        result = self.M.copy()
        for coefficient, (i, j) in zip(self.coefficients, self.pairs):
            base = coefficient * self.M[:, i] * self.M[:, j]
            result[:, i] += base * a[j]
            result[:, j] += base * a[i]
        return result


def fit_pairwise(Y, M, A, regularization):
    model = PairwiseModel(M, np.zeros(6))
    x = model.features(A).reshape(-1, 6)
    target = (Y - M @ A).ravel()
    # mean spectral squared error + regularization * mean(beta**2).
    augmented_x = np.vstack([x / np.sqrt(target.size), np.sqrt(regularization / 6) * np.eye(6)])
    augmented_y = np.r_[target / np.sqrt(target.size), np.zeros(6)]
    return np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
