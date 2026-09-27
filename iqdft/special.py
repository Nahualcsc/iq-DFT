"""Special functions not available (for complex arguments) in scipy."""
import numpy as np
from scipy.special import digamma  # noqa: F401  (re-exported; supports complex z)

# Bernoulli-number coefficients B_{2k} of the asymptotic expansion
# psi^(1)(w) ~ 1/w + 1/(2 w^2) + sum_k B_{2k} / w^{2k+1}
_B2K = (1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0,
        -691.0 / 2730.0, 7.0 / 6.0)
_SHIFT = 20


def trigamma(z):
    """Trigamma function psi^(1)(z) for complex z with Re(z) > 0 (vectorized).

    Uses the recurrence psi1(z) = psi1(z + N) + sum_{k<N} 1/(z+k)^2 to shift
    the argument to |w| > 20 and then the asymptotic (Stirling) series, which
    gives full double precision there.  In this code the argument is always
    z = 1/2 + (gamma/2 + i v)/(2 pi T), so Re(z) >= 1/2.
    """
    z = np.asarray(z, dtype=complex)
    if np.any(z.real <= 0):
        raise ValueError("trigamma implemented only for Re(z) > 0")
    acc = np.zeros_like(z)
    for k in range(_SHIFT):
        acc += 1.0 / (z + k) ** 2
    w = z + _SHIFT
    winv = 1.0 / w
    winv2 = winv * winv
    series = np.zeros_like(z)
    wpow = winv * winv2  # 1/w^3
    for b in _B2K:
        series += b * wpow
        wpow = wpow * winv2
    return acc + winv + 0.5 * winv2 + series
