from __future__ import annotations
import numpy as np
from dezero import Variable
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import cupy as cp  # type: ignore

gpu_enable: bool = True
try:
    import cupy as cp  # type: ignore
    cupy: Any|None = cp
    xpy: TypeAlias = cp
    xpndarray: TypeAlias = np.ndarray | cp.ndarray
except ImportError:
    gpu_enable = False
    cupy = None
    xpy: TypeAlias = np
    xpndarray: TypeAlias = np.ndarray


def get_array_module(x: xpndarray|Variable) -> xpy:
    if not gpu_enable:
        return np
    if isinstance(x, Variable):
        x = x.data
    return cp.get_array_module(x)

def as_numpy(x: xpndarray|Variable) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    if isinstance(x, Variable):
        x = x.data
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)

def as_cupy(x: xpndarray|Variable) -> cp.ndarray:
    if not gpu_enable:
        raise Exception(f"CuPy cannot be loaded. Install CuPy!")
    if isinstance(x, Variable):
        x = x.data
    return cp.asarray(x)