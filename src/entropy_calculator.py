"""
Shannon (1951) 不等式 (17) 的实现

给定排序概率 q_1 ≥ q_2 ≥ ... ≥ q_m（第 i 猜的频率）：

  H_lower = Σ (q_i - q_{i+1}) log₂ i    ← 下界
          = Σ p_i log₂ i                ← 等价形式, p_i = q_i
          = E[log₂ N], N = 猜中所需的次数

  H_mid   = -Σ q_i log₂ q_i            ← 标准香农熵 = 条件熵 F_N 的估计
  H_upper = H_mid                       ← 对理想预测器，上界紧贴 F_N

参考文献:
  Shannon (1951). Prediction and Entropy of Printed English.
  Bell System Technical Journal, 30, 50-64. Equation (17).
"""

import math


def lower_bound(probs):
    """H_lower = Σ p_i log₂ i = E[log₂ N]

    probs: 按概率降序排列 (p_1 ≥ p_2 ≥ ... ≥ p_m)
    """
    return sum(
        p * math.log2(rank)
        for rank, p in enumerate(probs, start=1)
        if p > 0 and rank > 1  # log(1)=0, 跳过
    )


def shannon_entropy(probs):
    """H = -Σ q_i log₂ q_i"""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def all_bounds(probs):
    """返回 (lower, middle)，upper 对理想预测器 ≡ middle"""
    return lower_bound(probs), shannon_entropy(probs)
