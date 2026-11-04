"""A simulated Earth: its magnetic field, its gravity, and what moves in them.

Normalized units: Earth's radius is 1, the dipole field at the magnetic equator's surface is 1, and
GM = 1. Nothing here is fitted; this module is the world the discovery engine (engine.py) is shown
measurements of, and must not read.

    field     B(r) = (3 (m . r^) r^ - m) / r^3                      a dipole, |m| = 1, tilted 11 degrees
                   + 5 (r^T Q r^) r^ / r^4 - 2 Q r^ / r^4             a quadrupole, B = -grad(r^T Q r / r^5)
              Q is symmetric and traceless; at the surface the quadrupole is about a tenth of the dipole.
              Outside the planet there are no currents, so div B = 0 and curl B = 0.
    gravity   a = -GM r / r^3
    charges   a = (q/m) v x B - GM r / r^3                            protons here have q/m = 50
              integrated with the Boris scheme (exact energy behaviour under the magnetic force)
"""
from __future__ import annotations

import numpy as np

TILT = np.radians(11.0)
M = np.array([-np.sin(TILT) * np.cos(np.radians(-72.0)), -np.sin(TILT) * np.sin(np.radians(-72.0)), -np.cos(TILT)])
Q = np.array([[0.04, 0.03, -0.02],
              [0.03, -0.07, 0.025],
              [-0.02, 0.025, 0.03]])          # symmetric, traceless
GM = 1.0
QM = 50.0


def field(r: np.ndarray) -> np.ndarray:
    """B at positions r, shape (..., 3)."""
    r = np.asarray(r, float)
    d = np.linalg.norm(r, axis=-1, keepdims=True)
    u = r / d
    dip = (3 * (u @ M)[..., None] * u - M) / d**3
    uqu = np.einsum("...i,ij,...j->...", u, Q, u)[..., None]
    quad = (5 * uqu * u - 2 * (u @ Q.T)) / d**4
    return dip + quad


def jacobian(r: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """dB_i/dx_j at r, shape (..., 3, 3), by central differences of the true field."""
    J = np.zeros(np.shape(r) + (3,))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        J[..., :, j] = (field(r + e) - field(r - e)) / (2 * h)
    return J


def accel(r, v, qm: float = 0.0, gravity: bool = True):
    a = np.zeros_like(r)
    if gravity:
        a -= GM * r / np.linalg.norm(r, axis=-1, keepdims=True) ** 3
    if qm:
        a += qm * np.cross(v, field(r))
    return a


def trajectory(r0, v0, qm: float, t_end: float, sample: float, substeps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Boris integration; returns sample times and positions (the only thing an observer records)."""
    dt = sample / substeps
    r, v = np.array(r0, float), np.array(v0, float)
    times, pos = [0.0], [r.copy()]
    for k in range(int(round(t_end / sample))):
        for _ in range(substeps):
            g = -GM * r / np.linalg.norm(r) ** 3
            v = v + 0.5 * dt * g
            if qm:
                t = 0.5 * dt * qm * field(r)
                s = 2 * t / (1 + t @ t)
                vp = v + np.cross(v, t)
                v = v + np.cross(vp, s)
            v = v + 0.5 * dt * g
            r = r + dt * v
        times.append((k + 1) * sample)
        pos.append(r.copy())
    return np.array(times), np.array(pos)


def survey(n: int, rmin: float, rmax: float, rng: np.random.Generator, noise: float = 0.0):
    """Magnetometer readings at n random points between rmin and rmax: positions and B, with relative
    noise (each component gets noise * |B| / sqrt(3) of Gaussian error)."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = u * rng.uniform(rmin, rmax, size=(n, 1))
    B = field(r)
    B = B + noise * np.linalg.norm(B, axis=1, keepdims=True) / np.sqrt(3) * rng.normal(size=B.shape)
    return r, B
