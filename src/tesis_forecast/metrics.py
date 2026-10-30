"""
Metricas compartidas.

Extraido verbatim (misma formula, mismo manejo de ceros/NaN) de la celda 49
del notebook legacy (funciones mape, smape, calcular_metricas). Esa misma
implementacion estaba duplicada de forma casi identica en varias celdas del
notebook (univariados, LSTM, SARIMAX, FCNN, Ensemble); aqui queda como la
unica version canonica para que los proximos modelos que se extraigan la
importen en vez de redefinirla.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calcular_metricas(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    y_c = np.array(y_true[:n], dtype=float)
    p_c = np.array(y_pred[:n], dtype=float)

    mask = (~np.isnan(y_c)) & (~np.isnan(p_c))
    if mask.sum() == 0:
        return None

    y_c = y_c[mask]
    p_c = p_c[mask]

    return {
        "MAE": mean_absolute_error(y_c, p_c),
        "RMSE": np.sqrt(mean_squared_error(y_c, p_c)),
        "MAPE": mape(y_c, p_c),
        "sMAPE": smape(y_c, p_c),
    }
