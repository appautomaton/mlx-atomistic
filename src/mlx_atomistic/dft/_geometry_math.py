"""Pure geometry-optimization vector mathematics shared by DFT workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _search_direction(
    gradient: np.ndarray,
    forces: np.ndarray,
    *,
    optimizer: str,
    s_history: Sequence[np.ndarray],
    y_history: Sequence[np.ndarray],
) -> np.ndarray:
    if optimizer == "steepest_descent" or not s_history:
        return forces.copy()
    direction = _lbfgs_direction(
        gradient.reshape(-1),
        s_history,
        y_history,
    ).reshape(forces.shape)
    if not np.isfinite(direction).all():
        return forces.copy()
    if float(np.sum(direction * forces)) <= 0.0:
        return forces.copy()
    return direction


def _lbfgs_direction(
    gradient: np.ndarray,
    s_history: Sequence[np.ndarray],
    y_history: Sequence[np.ndarray],
) -> np.ndarray:
    q = gradient.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for s_vector, y_vector in zip(
        reversed(s_history),
        reversed(y_history),
        strict=True,
    ):
        rho = 1.0 / float(np.dot(y_vector, s_vector))
        alpha = rho * float(np.dot(s_vector, q))
        q = q - alpha * y_vector
        alphas.append(alpha)
        rhos.append(rho)
    s_last = s_history[-1]
    y_last = y_history[-1]
    scale = float(np.dot(s_last, y_last) / max(np.dot(y_last, y_last), 1e-20))
    result = scale * q
    for s_vector, y_vector, alpha, rho in zip(
        s_history,
        y_history,
        reversed(alphas),
        reversed(rhos),
        strict=True,
    ):
        beta = rho * float(np.dot(y_vector, result))
        result = result + s_vector * (alpha - beta)
    return -result


def _valid_lbfgs_pair(s_vector: np.ndarray, y_vector: np.ndarray) -> bool:
    curvature = float(np.dot(s_vector, y_vector))
    return bool(np.isfinite(curvature) and curvature > 1e-12)


def _clip_direction(direction: np.ndarray, max_step: float) -> np.ndarray:
    clipped = np.array(direction, dtype=np.float64, copy=True)
    maximum = _max_vector_norm(clipped)
    if maximum > max_step:
        clipped *= max_step / maximum
    return clipped


def _max_vector_norm(values: np.ndarray) -> float:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(vectors.reshape((-1, 3)), axis=1)))


def _rms_components(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))
