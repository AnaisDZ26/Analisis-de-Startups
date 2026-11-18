"""Gráficos del simulador. Cada función devuelve un matplotlib.figure.Figure."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter

from . import escenario as E
from . import modelo as M
from .rubros import etiqueta, fmt_usd

VERDE, NARANJA, ROJO, AZUL, OSCURO, GRIS = "#1e8449", "#e67e22", "#c0392b", "#2c7fb8", "#2c3e50", "#7f8c8d"
TAM = {"titulo": 17, "eje": 14, "tick": 12, "leyenda": 12}


def _estilo(ax):
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=TAM["tick"])
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _pct(ax, eje="y"):
    fmt = FuncFormatter(lambda v, _: f"{100 * v:.0f} %".replace(".", ","))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(fmt)


def _sombrear_pocos_datos(ax, tramo, horizonte):
    ultimo = E.meses_con_datos(tramo)
    if ultimo < horizonte:
        ax.axvspan(ultimo, horizonte, color="#bdc3c7", alpha=0.25, zorder=0)
        ax.text(min(ultimo + 1, horizonte - 1), ax.get_ylim()[0], "  pocos datos observados:\n  proyección del modelo",
                fontsize=11, color=GRIS, va="bottom", ha="left")


# ---------------------------------------------------------------------------
def fig_supervivencia(modelo, rubro, monto, horizonte=120, conservador=False, mostrar_observado=True):
    """Curva S(t) de la startup configurada vs. la misma con US$ 100 mil."""
    t = np.linspace(1, horizonte, 240)
    k, theta = modelo["k"], modelo["rubros"][rubro]["theta_ajustada"]
    beta = M.beta_efectivo(modelo, conservador)
    x_tu, x_ref = float(M.x_desde_usd(monto)), float(M.x_desde_usd(E.MONTO_REFERENCIA))
    lo, med, hi = M.banda_supervivencia(t, modelo, rubro, x_tu, conservador=conservador)
    S_tu = M.supervivencia(t, theta, beta, x_tu, k)
    S_ref = M.supervivencia(t, theta, beta, x_ref, k)

    fig = Figure(figsize=(10, 5.6), layout="constrained")
    ax = fig.subplots()
    ax.fill_between(t, lo, hi, color=VERDE, alpha=0.18, label="Incertidumbre 95 %")
    if monto > E.MONTO_REFERENCIA:
        ax.fill_between(t, S_ref, S_tu, color=VERDE, alpha=0.28, label="Supervivencia ganada con el capital")
    ax.plot(t, S_ref, "--", color=GRIS, lw=2.2, label=f"Misma startup con {fmt_usd(E.MONTO_REFERENCIA)}")
    ax.plot(t, S_tu, color=VERDE, lw=3.2, label=f"Tu startup: {fmt_usd(monto)}")

    tramo = E.tramo_de(modelo, monto)
    if mostrar_observado:
        meses = np.array(tramo["meses"])
        m = meses <= horizonte
        ax.step(meses[m], np.array(tramo["S"])[m], where="post", color=OSCURO, lw=1.6, alpha=0.8,
                label=f"Observado (todos los rubros): {tramo['etiqueta']}")

    bajo = min(lo.min(), S_ref.min(), S_tu.min())
    ax.set_ylim(max(0.0, np.floor((bajo - 0.02) * 20) / 20), 1.005)
    ax.set_xlim(0, horizonte)
    _pct(ax)
    ax.set_xlabel("Edad de la startup (meses)", fontsize=TAM["eje"])
    ax.set_ylabel("Siguen abiertas", fontsize=TAM["eje"])
    ax.set_title(f"Curva de supervivencia · {etiqueta(rubro)}", fontsize=TAM["titulo"], fontweight="bold")
    _estilo(ax)
    _sombrear_pocos_datos(ax, tramo, horizonte)
    ax.legend(loc="lower left", fontsize=TAM["leyenda"], frameon=True)
    for h in (60, 120):
        if h <= horizonte:
            ax.axvline(h, color="#bdc3c7", lw=1, zorder=0)
    return fig


def fig_riesgo(modelo, rubro, monto, horizonte=120, conservador=False):
    """Función de riesgo h(t) en cierres por cada 10.000 startups en riesgo, por mes."""
    t = np.linspace(2, horizonte, 240)
    k, theta = modelo["k"], modelo["rubros"][rubro]["theta_ajustada"]
    beta = M.beta_efectivo(modelo, conservador)
    x_tu, x_ref = float(M.x_desde_usd(monto)), float(M.x_desde_usd(E.MONTO_REFERENCIA))
    lo, _, hi = M.banda_riesgo(t, modelo, rubro, x_tu, conservador=conservador)
    h_tu = M.funcion_riesgo(t, theta, beta, x_tu, k) * 1e4
    h_ref = M.funcion_riesgo(t, theta, beta, x_ref, k) * 1e4

    fig = Figure(figsize=(10, 5.2), layout="constrained")
    ax = fig.subplots()
    ax.fill_between(t, lo * 1e4, hi * 1e4, color=NARANJA, alpha=0.18, label="Incertidumbre 95 %")
    ax.plot(t, h_ref, "--", color=GRIS, lw=2.2, label=f"Con {fmt_usd(E.MONTO_REFERENCIA)}")
    ax.plot(t, h_tu, color=NARANJA, lw=3.2, label=f"Tu startup: {fmt_usd(monto)}")
    ax.set_ylim(0, max(hi.max() * 1e4, h_ref.max(), h_tu.max()) * 1.08)
    ax.set_xlim(0, horizonte)
    ax.set_xlabel("Edad de la startup (meses)", fontsize=TAM["eje"])
    ax.set_ylabel("Cierres por 10.000 startups\nen riesgo, por mes", fontsize=TAM["eje"])
    ax.set_title(f"Función de riesgo · {etiqueta(rubro)}", fontsize=TAM["titulo"], fontweight="bold")
    _estilo(ax)
    ax.legend(loc="upper left", fontsize=TAM["leyenda"], frameon=True)
    return fig


def fig_efecto_capital(modelo, rubro, monto, conservador=False):
    """Supervivencia a 5 y 10 años según el capital levantado (curva dosis-respuesta)."""
    montos = np.logspace(np.log10(2.5e4), np.log10(2.5e8), 120)
    x = M.x_desde_usd(montos)
    k, theta = modelo["k"], modelo["rubros"][rubro]["theta_ajustada"]
    beta = M.beta_efectivo(modelo, conservador)

    fig = Figure(figsize=(10, 5.2), layout="constrained")
    ax = fig.subplots()
    for h, color, nombre in ((120, AZUL, "10 años (120 m)"), (60, VERDE, "5 años (60 m)")):
        S = M.supervivencia(h, theta, beta, x, k)
        S_b = np.array([M.banda_supervivencia(np.array([h]), modelo, rubro, xi, n=300, conservador=conservador)[:, 0]
                        for xi in x[::6]])
        ax.fill_between(montos[::6], S_b[:, 0], S_b[:, 2], color=color, alpha=0.15)
        ax.plot(montos, S, color=color, lw=3, label=nombre)
        s_tu = float(M.supervivencia(h, theta, beta, float(M.x_desde_usd(monto)), k))
        ax.scatter([monto], [s_tu], s=140, color=color, edgecolor="black", zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(2.5e4, 2.5e8)
    ax.axvspan(1e8, 2.5e8, color="#bdc3c7", alpha=0.25, zorder=0)
    ax.text(1.12e8, ax.get_ylim()[0], "pocos datos (514 startups)", fontsize=11, color=GRIS,
            va="bottom", ha="left", rotation=90)
    ax.set_xticks([5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 2.5e8])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} mil"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    _pct(ax)
    ax.set_xlabel("Capital levantado en US$ (escala logarítmica; M = millones)", fontsize=TAM["eje"])
    ax.set_ylabel("Probabilidad de seguir abierta", fontsize=TAM["eje"])
    ax.set_title(f"Más capital, más supervivencia · {etiqueta(rubro)}", fontsize=TAM["titulo"], fontweight="bold")
    _estilo(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(0.92, 0.02), fontsize=TAM["leyenda"], frameon=True)
    return fig


def fig_waffle(S, horizonte_txt="5 años"):
    """100 puntos: cuántas de 100 startups como la tuya habrían cerrado."""
    cerradas = int(round((1 - S) * 100))
    fig = Figure(figsize=(4.6, 4.6), layout="constrained")
    ax = fig.subplots()
    idx = np.arange(100)
    fx, fy = idx % 10, 9 - idx // 10
    colores = [ROJO if i < cerradas else VERDE for i in idx]
    ax.scatter(fx, fy, s=330, c=colores, edgecolor="white", linewidth=1.5)
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(-0.7, 9.7)
    ax.axis("off")
    ax.set_title(f"100 startups como la tuya a {horizonte_txt}", fontsize=14, fontweight="bold")
    return fig


def fig_rubros(modelo, rubro_sel, monto, horizonte=120, conservador=False):
    """S(horizonte) ajustada de cada rubro con el capital elegido (los 18 más grandes + el elegido)."""
    beta = M.beta_efectivo(modelo, conservador)
    x = float(M.x_desde_usd(monto))
    filas = []
    for r, v in modelo["rubros"].items():
        S = float(M.supervivencia(horizonte, v["theta_ajustada"], beta, x, modelo["k"]))
        filas.append((r, S, v["n_financiadas"], v["cierres"], v["B"]))
    grandes = sorted(filas, key=lambda f: -f[2])[:18]
    elegidos = {f[0] for f in grandes} | {rubro_sel}
    datos = sorted([f for f in filas if f[0] in elegidos], key=lambda f: f[1])

    fig = Figure(figsize=(10, 0.42 * len(datos) + 1.8), layout="constrained")
    ax = fig.subplots()
    y = np.arange(len(datos))
    for yi, (r, S, n, c, B) in zip(y, datos):
        sel = r == rubro_sel
        ax.barh(yi, S, color=VERDE if sel else "#aab7b8", height=0.7, edgecolor="black" if sel else "none")
        ax.text(S + 0.003, yi, f"{100 * S:.1f} %".replace(".", ","), va="center", fontsize=11,
                fontweight="bold" if sel else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{etiqueta(f[0])}  ({f[3]} cierres)" for f in datos], fontsize=11)
    for lab, f in zip(ax.get_yticklabels(), datos):
        if f[0] == rubro_sel:
            lab.set_fontweight("bold")
    ax.set_xlim(max(0, min(f[1] for f in datos) - 0.05), 1.03)
    _pct(ax, "x")
    ax.set_xlabel(f"Probabilidad de seguir abierta a {horizonte} meses con {fmt_usd(monto)}", fontsize=TAM["eje"])
    ax.set_title("Mismo capital, distinto rubro", fontsize=TAM["titulo"], fontweight="bold")
    _estilo(ax)
    return fig
