"""In-place C = alpha * op(A) @ B + beta * C via the OpenBLAS bundled with
numpy (cblas_sgemm, 64-bit-int interface). numpy's matmul cannot accumulate
into its output; doing so saves one 400 KB pass per IA^2SGD event.
Falls back to numpy if the symbol cannot be found."""
import ctypes
import glob
import os

import numpy as np

_ROW, _NT, _T = 101, 111, 112
_fn = None
try:
    _libdir = os.path.join(os.path.dirname(os.path.dirname(np.__file__)), "numpy.libs")
    for _path in glob.glob(os.path.join(_libdir, "libscipy_openblas64_*.so")):
        _lib = ctypes.CDLL(_path)
        _fn = _lib.scipy_cblas_sgemm64_
        _fn.restype = None
        _i64, _i32, _f, _p = ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
        _fn.argtypes = [_i32, _i32, _i32, _i64, _i64, _i64, _f, _p, _i64, _p, _i64, _f, _p, _i64]
        break
except Exception:  # pragma: no cover
    _fn = None

HAVE_CBLAS = _fn is not None


def gemm_tn_acc(A, B, C, alpha, beta=1.0):
    """C <- alpha * A^T @ B + beta * C (float32, C-contiguous, in place).
    A: (K, M), B: (K, N), C: (M, N)."""
    K, M = A.shape
    K2, N = B.shape
    assert K == K2 and C.shape == (M, N)
    if _fn is None or not (A.flags.c_contiguous and B.flags.c_contiguous and C.flags.c_contiguous):
        C *= np.float32(beta)
        C += np.float32(alpha) * (A.T @ B)
        return C
    assert A.dtype == B.dtype == C.dtype == np.float32
    _fn(_ROW, _T, _NT, M, N, K, float(alpha), A.ctypes.data, M, B.ctypes.data, N, float(beta), C.ctypes.data, N)
    return C
