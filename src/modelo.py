"""
Núcleo estadístico del simulador (solo numpy / scipy).

Modelo (extensión del Método 3 del proyecto, con el capital como covariable):

    H_j(t | x) = exp(theta_j + beta * x) * (t / 120) ** k        (riesgo acumulado)
    S_j(t | x) = exp(-H_j(t | x))                                (supervivencia)
    h_j(t | x) = k / t * H_j(t | x)                              (función de riesgo)

    x       = log10(raised_amount_usd) - 6      (x = 0 equivale a US$ 1 millón)
    k       = forma Weibull común a todos los rubros
    beta    = efecto del capital sobre el log-riesgo (por cada 10x más dinero)
    theta_j = log del riesgo acumulado a 120 meses del rubro j con x = 0;
              se estima por Bayes empírico (DerSimonian-Laird), como en el Método 3.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import approx_fprime, minimize
from scipy.stats import chi2

HORIZONTE_REF = 120          # meses (10 años), igual que el notebook
LOG10_REF = 6.0              # x = log10(USD) - 6


def x_desde_usd(monto_usd):
    """Transforma un monto en dólares a la escala x del modelo."""
    return np.log10(np.asarray(monto_usd, dtype=float)) - LOG10_REF


# ---------------------------------------------------------------------------
# 1) Ajuste Weibull con efectos de rubro por verosimilitud perfilada
# ---------------------------------------------------------------------------
def _nll_perfilada(theta, t, c, logt, grupo, X, n_grupos):
    """Log-verosimilitud negativa Weibull PH con un nivel base por rubro.

    Con (k, beta) fijos, el MLE del nivel de cada rubro tiene forma cerrada
    (d_j / sum(exp(X beta) t^k)); por eso se "perfila" y solo se optimiza (log k, beta).
    """
    k = np.exp(theta[0])
    lin = X @ theta[1:] if X.shape[1] else np.zeros(len(t))
    expo = np.bincount(grupo, weights=np.exp(lin) * t ** k, minlength=n_grupos)
    fallas = np.bincount(grupo, weights=c, minlength=n_grupos)
    ok = fallas > 0
    ll = np.sum(c * (np.log(k) + (k - 1) * logt + lin))
    ll += np.sum(fallas[ok] * np.log(fallas[ok] / expo[ok]) - fallas[ok])
    return -ll


def ajustar_weibull_perfilado(t, c, grupo, X, n_grupos):
    """MLE de (log k, beta) con rubros como efectos fijos. Devuelve parámetros y covarianza."""
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    X = np.asarray(X, float).reshape(len(t), -1)
    logt = np.log(t)
    args = (t, c, logt, grupo, X, n_grupos)
    th0 = np.zeros(1 + X.shape[1])
    th0[0] = np.log(0.9)
    with np.errstate(all="ignore"):   # BFGS prueba valores extremos al inicio; no afecta el óptimo
        res = minimize(_nll_perfilada, th0, args=args, method="BFGS", options={"gtol": 1e-6})

    # Hessiana numérica (diferencias centrales del gradiente numérico)
    n = len(res.x)
    H = np.zeros((n, n))
    eps = 1e-4
    grad = lambda z: approx_fprime(z, _nll_perfilada, 1e-6, *args)
    with np.errstate(all="ignore"):
        for i in range(n):
            e = np.zeros(n)
            e[i] = eps
            H[:, i] = (grad(res.x + e) - grad(res.x - e)) / (2 * eps)
    cov = np.linalg.inv((H + H.T) / 2)
    return {"log_k": float(res.x[0]), "k": float(np.exp(res.x[0])),
            "beta": res.x[1:].copy(), "cov": cov, "nll": float(res.fun),
            "convergio": bool(res.success)}


# ---------------------------------------------------------------------------
# 2) Bayes empírico por rubro (mismo esquema que el Método 3 del notebook)
# ---------------------------------------------------------------------------
def bayes_empirico_normal(estimaciones, varianzas):
    """Modelo normal-normal con DerSimonian-Laird sobre el log del riesgo acumulado."""
    J = len(estimaciones)
    w = 1 / varianzas
    media_fija = np.sum(w * estimaciones) / np.sum(w)
    Q = np.sum(w * (estimaciones - media_fija) ** 2)
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (J - 1)) / denom)
    w_re = 1 / (varianzas + tau2)
    mu = np.sum(w_re * estimaciones) / np.sum(w_re)
    var_mu = 1 / np.sum(w_re)
    B = varianzas / (varianzas + tau2)
    ajustada = B * mu + (1 - B) * estimaciones
    var_post = (1 - B) * varianzas + B ** 2 * var_mu
    return {"mu": float(mu), "tau2": float(tau2), "Q": float(Q), "p_Q": float(chi2.sf(Q, J - 1)),
            "I2": float(max(0.0, (Q - (J - 1)) / Q)), "B": B, "theta_ajustada": ajustada,
            "sd_posterior": np.sqrt(var_post)}


def estimar_rubros(t, c, grupo, x, k, beta, n_grupos):
    """Estimación cruda (forma cerrada con k y beta fijos) y ajustada de cada rubro."""
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    x = np.asarray(x, float)
    exposicion = np.bincount(grupo, weights=np.exp(beta * x) * t ** k, minlength=n_grupos)
    cierres = np.bincount(grupo, weights=c, minlength=n_grupos)
    cierres_corr = np.where(cierres == 0, 0.5, cierres)      # corrección 0.5, como en el notebook
    theta_crudo = np.log(HORIZONTE_REF ** k * cierres_corr / exposicion)
    eb = bayes_empirico_normal(theta_crudo, 1 / cierres_corr)
    return {"cierres": cierres.astype(int), "theta_crudo": theta_crudo, **eb}


# ---------------------------------------------------------------------------
# 3) Funciones del modelo
# ---------------------------------------------------------------------------
def riesgo_acumulado(t, theta, beta, x, k):
    return np.exp(theta + beta * x) * (np.asarray(t, float) / HORIZONTE_REF) ** k


def supervivencia(t, theta, beta, x, k):
    return np.exp(-riesgo_acumulado(t, theta, beta, x, k))


def funcion_riesgo(t, theta, beta, x, k):
    """h(t): probabilidad instantánea de cierre por mes."""
    t = np.asarray(t, float)
    return k / t * riesgo_acumulado(t, theta, beta, x, k)


def tiempo_hasta_fraccion_cerrada(p, theta, beta, x, k):
    """Mes en que habrá cerrado la fracción p de empresas con ese perfil."""
    return HORIZONTE_REF * (-np.log(1 - p) / np.exp(theta + beta * x)) ** (1 / k)


# ---------------------------------------------------------------------------
# 4) Incertidumbre (bandas al 95 %)
# ---------------------------------------------------------------------------
def simular_parametros(modelo: dict, rubro: str, n=600, semilla=2026, conservador=False):
    """Muestras aproximadas de (k, beta, theta_rubro).

    Escenario base: (log k, beta) ~ Normal multivariada con la covarianza del ajuste.
    Escenario conservador: beta ~ Normal(beta del análisis landmark, su EE), independiente de k.
    En ambos: theta_rubro ~ Normal(theta ajustada, desviación posterior del Bayes empírico).
    """
    rng = np.random.default_rng(semilla)
    cov = np.array(modelo["cov_logk_beta"])
    if conservador:
        rob = modelo["robustez_landmark"]
        log_k = rng.normal(np.log(modelo["k"]), np.sqrt(cov[0, 0]), size=n)
        beta = rng.normal(rob["beta"], rob["se_beta"], size=n)
    else:
        lk_beta = rng.multivariate_normal([np.log(modelo["k"]), modelo["beta"]], cov, size=n)
        log_k, beta = lk_beta[:, 0], lk_beta[:, 1]
    r = modelo["rubros"][rubro]
    theta = rng.normal(r["theta_ajustada"], r["sd_posterior"], size=n)
    return np.exp(log_k), beta, theta


def beta_efectivo(modelo: dict, conservador=False):
    return modelo["robustez_landmark"]["beta"] if conservador else modelo["beta"]


def banda_supervivencia(t, modelo, rubro, x, n=600, conservador=False):
    """Percentiles 2.5 / 50 / 97.5 de S(t) a partir de las muestras de parámetros."""
    k, b, th = simular_parametros(modelo, rubro, n, conservador=conservador)
    t = np.asarray(t, float)[None, :]
    S = np.exp(-np.exp(th[:, None] + b[:, None] * x) * (t / HORIZONTE_REF) ** k[:, None])
    return np.percentile(S, [2.5, 50, 97.5], axis=0)


def banda_riesgo(t, modelo, rubro, x, n=600, conservador=False):
    """Percentiles 2.5 / 50 / 97.5 de h(t)."""
    k, b, th = simular_parametros(modelo, rubro, n, conservador=conservador)
    t = np.asarray(t, float)[None, :]
    H = np.exp(th[:, None] + b[:, None] * x) * (t / HORIZONTE_REF) ** k[:, None]
    h = k[:, None] / t * H
    return np.percentile(h, [2.5, 50, 97.5], axis=0)
