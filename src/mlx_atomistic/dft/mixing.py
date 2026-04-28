"""Density mixers for the DFT SCF loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class LinearMixer:
    """Simple linear density mixing."""

    beta: float = 0.35
    name: str = "linear"

    def __post_init__(self) -> None:
        if not 0.0 < self.beta <= 1.0:
            msg = "linear mixing beta must be in the interval (0, 1]"
            raise ValueError(msg)

    def reset(self) -> None:
        """Reset mixer state."""

    def mix(self, current: mx.array, target: mx.array) -> mx.array:
        """Return a mixed density."""

        return (1.0 - self.beta) * current + self.beta * target

    def metadata(self) -> dict[str, float | int | str]:
        """Return JSON-safe mixer metadata."""

        return {"name": self.name, "beta": self.beta}


@dataclass
class PulayDIISMixer:
    """Pulay DIIS mixer for density residuals."""

    beta: float = 0.35
    history_size: int = 6
    regularization: float = 1e-10
    name: str = "pulay-diis"

    def __post_init__(self) -> None:
        if not 0.0 < self.beta <= 1.0:
            msg = "DIIS beta must be in the interval (0, 1]"
            raise ValueError(msg)
        if self.history_size < 2:
            msg = "DIIS history_size must be at least 2"
            raise ValueError(msg)
        if self.regularization < 0.0:
            msg = "DIIS regularization must be non-negative"
            raise ValueError(msg)
        self._densities: list[np.ndarray] = []
        self._residuals: list[np.ndarray] = []
        self._last_coefficients: list[float] = []

    def reset(self) -> None:
        """Clear DIIS history."""

        self._densities.clear()
        self._residuals.clear()
        self._last_coefficients = []

    def mix(self, current: mx.array, target: mx.array) -> mx.array:
        """Return a DIIS-mixed density, falling back to linear mixing early on."""

        linear = (1.0 - self.beta) * current + self.beta * target
        residual = np.array(target - current, dtype=np.float64).reshape(-1)
        candidate = np.array(linear, dtype=np.float64).reshape(-1)
        self._densities.append(candidate)
        self._residuals.append(residual)
        if len(self._densities) > self.history_size:
            self._densities.pop(0)
            self._residuals.pop(0)
        if len(self._densities) < 2:
            self._last_coefficients = [1.0]
            return linear

        count = len(self._residuals)
        matrix = np.empty((count + 1, count + 1), dtype=np.float64)
        matrix[:count, :count] = [
            [float(np.dot(left, right)) for right in self._residuals]
            for left in self._residuals
        ]
        matrix[:count, :count] += self.regularization * np.eye(count)
        matrix[:count, count] = -1.0
        matrix[count, :count] = -1.0
        matrix[count, count] = 0.0
        rhs = np.zeros(count + 1, dtype=np.float64)
        rhs[count] = -1.0
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            self._last_coefficients = [1.0]
            return linear
        coefficients = solution[:count]
        mixed = np.sum(
            [
                coefficient * density
                for coefficient, density in zip(coefficients, self._densities, strict=True)
            ],
            axis=0,
        )
        self._last_coefficients = [float(value) for value in coefficients]
        return mx.array(mixed.reshape(current.shape).astype(np.float32))

    def metadata(self) -> dict[str, float | int | str | Sequence[float]]:
        """Return JSON-safe mixer metadata."""

        return {
            "name": self.name,
            "beta": self.beta,
            "history_size": self.history_size,
            "stored": len(self._densities),
            "last_coefficients": list(self._last_coefficients),
        }
