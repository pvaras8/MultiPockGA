from typing import Callable, Dict

import numpy as np
from numpy import exp


def sigmoid_pen_docking(x, min_x=-12.0, max_x=-6.0, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def sigmoid_pen_weight(x, min_x=300.0, max_x=500, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def combine_docking_only(d):
    return 1 - sigmoid_pen_docking(d)


def combine_docking_multi(docking_values, aggregation="mean"):
    values = np.asarray(docking_values, dtype=float)
    if values.size == 0:
        raise ValueError("No hay valores de docking para combinar")

    if aggregation == "mean":
        agg_value = float(np.mean(values))
    elif aggregation == "min":
        agg_value = float(np.min(values))
    elif aggregation == "max":
        agg_value = float(np.max(values))
    else:
        raise ValueError(f"Agregacion de docking no soportada: {aggregation}")

    return 1 - sigmoid_pen_docking(agg_value)


def combine_docking_logP(docking, logp):
    def docking_component(d, d_min=-10.0, d_max=-4.0, s_min=-6.0, s_max=6.0, power=1.3):
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


def combine_weighted_sum(values, weights, bias=0.0):
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if arr.shape[0] != w.shape[0]:
        raise ValueError("values y weights deben tener la misma longitud")
    return float(np.dot(arr, w) + float(bias))


COMBINERS: Dict[str, Callable] = {
    "docking_only": combine_docking_only,
    "docking_multi": combine_docking_multi,
    "docking_logp": combine_docking_logP,
    "mw_only": sigmoid_pen_weight,
    "weighted_sum": combine_weighted_sum,
}