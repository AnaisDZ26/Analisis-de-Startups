"""Pruebas básicas del modelo. Ejecutar:  python -m pytest -q   (o  python tests/test_modelo.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import escenario as E  # noqa: E402
from src import modelo as M  # noqa: E402

modelo = E.cargar_modelo()


def test_mas_capital_mas_supervivencia():
    for rubro in ("software", "web", "biotech"):
        S = [E.calcular(modelo, rubro, m)["tu"]["S120"] for m in (5e4, 5e5, 5e6, 5e7)]
        assert all(a < b for a, b in zip(S, S[1:])), rubro


def test_curvas_validas():
    t = np.linspace(1, 120, 100)
    S = M.supervivencia(t, -2.0, modelo["beta"], 0.3, modelo["k"])
    assert np.all((S > 0) & (S <= 1)) and np.all(np.diff(S) < 0)


def test_hazard_coherente_con_supervivencia():
    # h(t) = -d/dt log S(t)
    t, dt = 60.0, 1e-4
    th, b, x, k = -2.0, modelo["beta"], 0.3, modelo["k"]
    num = -(np.log(M.supervivencia(t + dt, th, b, x, k)) - np.log(M.supervivencia(t - dt, th, b, x, k))) / (2 * dt)
    assert abs(num - M.funcion_riesgo(t, th, b, x, k)) < 1e-6


def test_percentil_inverso():
    th, b, x, k = -2.0, modelo["beta"], 0.3, modelo["k"]
    t5 = M.tiempo_hasta_fraccion_cerrada(0.05, th, b, x, k)
    assert abs(M.supervivencia(t5, th, b, x, k) - 0.95) < 1e-9


def test_bayes_empirico_contrae_mas_los_rubros_con_pocos_cierres():
    rub = modelo["rubros"]
    pocos = [v["B"] for v in rub.values() if v["cierres"] < 10]
    muchos = [v["B"] for v in rub.values() if v["cierres"] >= 100]
    assert np.mean(pocos) > np.mean(muchos)
    assert all(0 <= v["B"] <= 1 for v in rub.values())


def test_bandas_contienen_estimacion_puntual():
    r = E.calcular(modelo, "software", 2e6)
    lo, hi = r["tu"]["S120_ic"]
    assert lo < r["tu"]["S120"] < hi


def test_ajuste_recupera_parametros_simulados():
    rng = np.random.default_rng(1)
    n, J = 40000, 5
    g = rng.integers(0, J, n)
    x = rng.normal(0, 1, n)
    k0, b0 = 1.3, -0.6
    lam = np.exp(rng.normal(-5, 0.3, J))[g]
    u = rng.uniform(size=n)
    t = (-np.log(u) / (lam * np.exp(b0 * x))) ** (1 / k0)
    cens = rng.uniform(20, 120, n)
    c = (t <= cens).astype(float)
    t = np.minimum(t, cens)
    aj = M.ajustar_weibull_perfilado(t, c, g, x[:, None], J)
    assert abs(aj["k"] - k0) < 0.05 and abs(aj["beta"][0] - b0) < 0.05


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("ok ", nombre)
