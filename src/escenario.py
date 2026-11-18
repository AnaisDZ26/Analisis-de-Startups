"""Cálculo de un escenario (rubro + monto) listo para mostrar en la interfaz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import modelo as M

RUTA_MODELO = Path(__file__).resolve().parents[1] / "data" / "modelo.json"
MONTO_REFERENCIA = 100_000          # contraste: startup con solo US$ 100 mil
FRACCION_CIERRE = 0.05              # "¿en qué mes habrá cerrado el 5 % de startups como la tuya?"
TOPE_MESES = 120                    # más allá se habla de "más de 10 años" (extrapolación)


def cargar_modelo(ruta: Path = RUTA_MODELO) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _percentiles_en(modelo, rubro, x, t, conservador):
    S = M.banda_supervivencia(np.array([t]), modelo, rubro, x, conservador=conservador)
    return float(S[0, 0]), float(S[1, 0]), float(S[2, 0])     # (p2.5, mediana, p97.5)


def calcular(modelo: dict, rubro: str, monto_usd: float, conservador: bool = False) -> dict:
    """Todas las cifras que muestra la app para (rubro, monto)."""
    k = modelo["k"]
    beta = M.beta_efectivo(modelo, conservador)
    theta = modelo["rubros"][rubro]["theta_ajustada"]
    out = {"k": k, "beta": beta, "theta": theta}
    for etiqueta, monto in (("tu", monto_usd), ("ref", MONTO_REFERENCIA)):
        x = float(M.x_desde_usd(monto))
        d = {"x": x}
        for h in (60, 120):
            d[f"S{h}"] = float(M.supervivencia(h, theta, beta, x, k))
            lo, _, hi = _percentiles_en(modelo, rubro, x, h, conservador)
            d[f"S{h}_ic"] = (lo, hi)
        t_p = float(M.tiempo_hasta_fraccion_cerrada(FRACCION_CIERRE, theta, beta, x, k))
        d["t_cierre"] = t_p
        d["t_cierre_fuera"] = t_p > TOPE_MESES
        out[etiqueta] = d
    out["hr_vs_ref"] = float(np.exp(beta * (out["tu"]["x"] - out["ref"]["x"])))
    return out


def tramo_de(modelo: dict, monto_usd: float) -> dict:
    """Tramo de capital (Kaplan-Meier observado) al que pertenece el monto."""
    limites = [1e5, 1e6, 1e7, 1e8]
    i = int(np.digitize(monto_usd, limites, right=True))
    return modelo["km_tramos"][i]


def meses_con_datos(tramo: dict, minimo_en_riesgo: int = 100) -> int:
    """Último mes con al menos `minimo_en_riesgo` empresas observadas en el tramo."""
    ok = [m for m, n in zip(tramo["meses"], tramo["en_riesgo"]) if n >= minimo_en_riesgo]
    return max(ok) if ok else 0
