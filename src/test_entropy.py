"""
entropy_calculator 单元测试

用法:
  python pj/src/test_entropy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from entropy_calculator import all_bounds, shannon_entropy, lower_bound


def test_deterministic():
    p = [1.0]
    l, m = all_bounds(p)
    assert abs(l - 0.0) < 1e-9
    assert abs(m - 0.0) < 1e-9
    print("  ✓ deterministic (H=0)")


def test_uniform():
    m = 26
    p = [1.0 / m] * m
    l, mid = all_bounds(p)
    expected = -m * (1.0 / m) * __import__("math").log2(1.0 / m)
    assert abs(mid - expected) < 1e-9
    assert l <= mid, f"lower {l} > middle {mid}"
    print(f"  ✓ uniform m={m}: mid={mid:.4f}, lower={l:.4f}")


def test_skewed():
    p = [0.7, 0.2, 0.1]
    l, m = all_bounds(p)
    # H_lower = 0.7*log(1) + 0.2*log(2) + 0.1*log(3) = 0.2 + 0.1585 = 0.3585
    expected_lower = 0.2 * 1.0 + 0.1 * __import__("math").log2(3)
    assert abs(l - expected_lower) < 1e-9
    assert l <= m, f"lower {l} > middle {m}"
    print(f"  ✓ skewed: mid={m:.4f}, lower={l:.4f}, gap={m-l:.4f}")


def test_concentrated():
    """高度集中分布：下界应接近中值"""
    p = [0.95, 0.03, 0.01, 0.01]
    l, m = all_bounds(p)
    assert l <= m
    gap = m - l
    print(f"  ✓ concentrated: mid={m:.4f}, lower={l:.4f}, gap={gap:.4f}")
    assert gap < 1.0, f"gap too large: {gap}"


if __name__ == "__main__":
    test_deterministic()
    test_uniform()
    test_skewed()
    test_concentrated()
    print("\n✅ All tests passed")
