import numpy as np

from scripts.analyze_svd_diagonal import analyze


def _synthetic_band(n=400, floor=1.0 / 400, amp=0.5, p=1.0, ell=8.0):
    """R[j,k] = floor + amp * (j+1)^{-p} * exp(-|k-j|/ell): a diagonal band with a known
    longitudinal power law and exponential transverse profile."""
    j = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    return floor + amp * (j + 1.0) ** (-p) * np.exp(-np.abs(k - j) / ell)


def test_longitudinal_power_law_recovered():
    res = analyze(_synthetic_band(p=1.0), max_offset=60, transverse_smooth=2)
    assert abs(res['long_fit']['slope'] + 1.0) < 0.15   # A(i)-floor ~ i^-1
    assert res['long_fit']['r2'] > 0.9


def test_transverse_shape_is_exponential():
    res = analyze(_synthetic_band(ell=8.0), max_offset=60, transverse_smooth=2)
    verdicts = [sh['best'] for sh in res['shapes'].values() if sh['best'] != 'undetermined']
    assert verdicts, 'no resolvable transverse cross-sections'
    # Exponential should win at the majority of probe positions.
    assert verdicts.count('exponential') >= (len(verdicts) + 1) // 2
