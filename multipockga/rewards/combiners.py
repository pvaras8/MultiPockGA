from typing import Callable, Dict

import numpy as np
from numpy import exp


# ============================================================
# BASE: DOCKING -> [0,1]
# ============================================================

def sigmoid_pen_docking(x, min_x=-12.0, max_x=-6.0, min_sig_in=-6.0, max_sig_in=6.0):
    sig = lambda y: 1.0 / (1.0 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(float(x)))


def docking_reward(d):
    # más negativo = mejor → reward alta
    return 1.0 - sigmoid_pen_docking(float(d))


# ============================================================
# DOCKING SIMPLE
# ============================================================

def combine_docking_only(d):
    return docking_reward(d)


def combine_two_docking(d1, d2, alpha=0.5, beta=0.5):
    total = float(alpha) + float(beta)
    if total <= 0:
        raise ValueError("alpha + beta debe ser mayor que 0")

    alpha_n = float(alpha) / total
    beta_n = float(beta) / total

    r1 = docking_reward(d1)
    r2 = docking_reward(d2)

    return float(alpha_n * r1 + beta_n * r2)


# ============================================================
# QED
# ============================================================

def qed_component(qed, threshold=0.5, sharpness=10.0, floor=0.2):
    """
    Factor en [floor, 1]
    """
    s = 1.0 / (1.0 + np.exp(-sharpness * (float(qed) - threshold)))
    return float(floor + (1.0 - floor) * s)


def combine_two_docking_qed(
    d1,
    d2,
    qed,
    alpha=0.5,
    beta=0.5,
    eps_floor=0.02,
):
    total = float(alpha) + float(beta)
    if total <= 0:
        raise ValueError("alpha + beta debe ser mayor que 0")

    alpha_n = alpha / total
    beta_n = beta / total

    r1 = docking_reward(d1)
    r2 = docking_reward(d2)

    rdock = (r1 ** alpha_n) * (r2 ** beta_n)
    rqed = qed_component(qed)

    return float(np.clip(rdock * rqed, eps_floor, 1.0))


# ============================================================
# LOGP
# ============================================================

def combine_docking_logP(docking, logp):
    def docking_component(d, d_min=-12.0, d_max=-6.0, s_min=-6.0, s_max=6.0, power=1.3):
        z = (d - d_min) / (d_max - d_min) * (s_max - s_min) + s_min
        base = 1.0 - (1.0 / (1.0 + np.exp(-z)))
        return np.clip(base, 0.0, 1.0) ** power

    def logp_component(lp, mu=2.5, sigma=1.0):
        return np.exp(-((lp - mu) ** 2) / (2.0 * sigma**2))

    d_comp = docking_component(docking)
    lp_comp = logp_component(logp)

    w_lp = 0.5
    bias = 0.2
    eps_floor = 0.02

    mod = (bias + w_lp * lp_comp) / (bias + w_lp)
    reward = np.clip(d_comp * mod, eps_floor, 1.0)

    return float(reward)


def combine_two_docking_logP(
    d1,
    d2,
    logp,
    alpha=0.5,
    beta=0.5,
    mu=2.5,
    sigma=1.0,
    eps_floor=0.02,
):
    total = float(alpha) + float(beta)
    if total <= 0:
        raise ValueError("alpha + beta debe ser mayor que 0")

    alpha_n = float(alpha) / total
    beta_n = float(beta) / total

    r1 = docking_reward(d1)
    r2 = docking_reward(d2)

    docking_part = (r1 ** alpha_n) * (r2 ** beta_n)
    logp_part = np.exp(-((float(logp) - float(mu)) ** 2) / (2.0 * float(sigma) ** 2))

    reward = np.clip(docking_part * logp_part, float(eps_floor), 1.0)
    return float(reward)


# ============================================================
# REGISTRO
# ============================================================

COMBINERS: Dict[str, Callable] = {
    "docking_only": combine_docking_only,
    "docking_two": combine_two_docking,
    "docking_logp": combine_docking_logP,
    "docking_two_logp": combine_two_docking_logP,
    "docking_two_qed": combine_two_docking_qed,
}