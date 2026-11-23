"""Strict JSON helpers for interoperable ML artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


def json_safe(value: object) -> object:
    """Recursively replace non-JSON numeric values with ``null``.

    Estimator parameter dictionaries may contain NumPy scalars and NaN (for
    example XGBoost's default ``missing`` value). JSON permits neither NumPy
    scalar objects nor non-finite number tokens in its interoperable grammar.
    """
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def strict_json_dumps(value: object, **kwargs: Any) -> str:
    """Serialize standards-compliant JSON and reject residual non-finite values."""
    return json.dumps(json_safe(value), allow_nan=False, **kwargs)
