"""Small expansion-model helpers for comoving-coordinate simulations."""
from __future__ import annotations

import math

_MIN_A = 1.0e-12


def scale_factor(t: float, H0: float, model: str = "linear") -> float:
    """Return a(t) for the selected expansion model.

    Supported models:
      - none:        a(t) = 1
      - linear:      a(t) = 1 + H0*t
      - exponential: a(t) = exp(H0*t)
    """
    model = (model or "none").lower()
    if model == "none" or H0 == 0:
        return 1.0
    if model == "linear":
        return max(_MIN_A, 1.0 + H0 * t)
    if model == "exponential":
        # Avoid overflow in extremely long experiments.
        return max(_MIN_A, math.exp(max(-700.0, min(700.0, H0 * t))))
    raise ValueError(f"Unknown expansion model: {model}")


def expansion_rate(t: float, H0: float, model: str = "linear") -> float:
    """Return H(t) = a_dot/a for the selected expansion model."""
    model = (model or "none").lower()
    if model == "none" or H0 == 0:
        return 0.0
    if model == "linear":
        return H0 / scale_factor(t, H0, model)
    if model == "exponential":
        return H0
    raise ValueError(f"Unknown expansion model: {model}")
