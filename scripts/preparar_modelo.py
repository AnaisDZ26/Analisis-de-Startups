"""
Prepara el modelo del simulador a partir del dataset original de Crunchbase.

Uso (una sola vez, en tu notebook, con el dataset de Kaggle):

    python scripts/preparar_modelo.py --datos ruta/a/dataset.zip
    python scripts/preparar_modelo.py --datos ruta/a/carpeta_con_los_csv

Genera data/modelo.json (~50 KB). La app SOLO necesita ese archivo, por eso se puede
desplegar en Streamlit Community Cloud sin subir los 285 MB de objects.csv.

Replica el pipeline del notebook Proyecto_Estadistica (misma muestra: 91,206 empresas).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import modelo as M  # noqa: E402

FECHA_CORTE = pd.Timestamp("2013-12-31")
LANDMARK_MESES = 12
TRAMOS_USD = [0, 1e5, 1e6, 1e7, 1e8, np.inf]
ETIQUETAS_TRAMOS = ["menos de US$ 100 mil", "US$ 100 mil – 1 millón", "US$ 1 – 10 millones",
                    "US$ 10 – 100 millones", "más de US$ 100 millones"]
HORIZONTE_KM = 120


# ---------------------------------------------------------------------------
def leer_csv(origen: Path, nombre: str, usecols):
    if origen.suffix == ".zip":
        with zipfile.ZipFile(origen) as z, z.open(nombre) as f:
            return pd.read_csv(f, usecols=usecols, low_memory=False)
    return pd.read_csv(origen / nombre, usecols=usecols, low_memory=False)


def construir_muestra(origen: Path) -> pd.DataFrame:
    """Pipeline idéntico al del notebook (secciones 2-4)."""
    obj = leer_csv(origen, "objects.csv",
                   ["id", "entity_type", "category_code", "status", "founded_at", "closed_at"])
    acq = leer_csv(origen, "acquisitions.csv", ["acquired_object_id", "acquired_at"])
    ipo = leer_csv(origen, "ipos.csv", ["object_id", "public_at"])
    fr = leer_csv(origen, "funding_rounds.csv", ["object_id", "funded_at", "raised_amount_usd"])

    df = obj[obj["entity_type"] == "Company"].copy()
    df = df.merge(acq, left_on="id", right_on="acquired_object_id", how="left")
    df = df.merge(ipo, left_on="id", right_on="object_id", how="left")
    fondos = fr.groupby("object_id")["raised_amount_usd"].sum().reset_index()
    df = df.merge(fondos, left_on="id", right_on="object_id", how="left", suffixes=("", "_f"))
    df["raised_amount_usd"] = df["raised_amount_usd"].fillna(0)

    for col in ["founded_at", "closed_at", "acquired_at", "public_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["founded_at"])

    fecha_final = df["closed_at"].fillna(df["public_at"]).fillna(df["acquired_at"]).fillna(FECHA_CORTE)
    df["T_meses"] = (fecha_final - df["founded_at"]).dt.days / 30.44
    df = df[df["T_meses"] > 0].copy()
    df["C_falla"] = (df["status"] == "closed").astype(int)

    # capital levantado dentro de los primeros LANDMARK_MESES (para el chequeo de robustez)
    m = fr.assign(funded_at=pd.to_datetime(fr["funded_at"], errors="coerce")) \
          .merge(df[["id", "founded_at"]].drop_duplicates("id"), left_on="object_id", right_on="id")
    m["mes"] = (m["funded_at"] - m["founded_at"]).dt.days / 30.44
    temprano = m[m["mes"] <= LANDMARK_MESES].groupby("id")["raised_amount_usd"].sum()
    df["raised_temprano_usd"] = df["id"].map(temprano).fillna(0)
    return df


def km(t, c):
    """Kaplan-Meier con numpy: devuelve tiempos únicos y S(t)."""
    u, inv = np.unique(t, return_inverse=True)
    d = np.bincount(inv, weights=c)
    m = np.bincount(inv)
    en_riesgo = len(t) - np.concatenate([[0], np.cumsum(m)[:-1]])
    return u, np.cumprod(1 - d / en_riesgo)


def km_en(u, S, p):
    i = np.searchsorted(u, p, side="right") - 1
    return 1.0 if i < 0 else float(S[i])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, type=Path, help="dataset.zip o carpeta con los CSV")
    ap.add_argument("--salida", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "modelo.json")
    args = ap.parse_args()

    print("Construyendo muestra (puede tardar ~1 min por objects.csv)...")
    df = construir_muestra(args.datos)
    con_rubro = df.dropna(subset=["category_code"]).copy()
    print(f"  empresas: {len(df):,} | con rubro: {len(con_rubro):,} | cierres: {int(df['C_falla'].sum()):,}")

    # --- Descriptivo: financiadas vs no financiadas (hallazgo de sub-registro) -----------------
    fin_mask = con_rubro["raised_amount_usd"] > 0
    desc = {
        "n_sin_inversion": int((~fin_mask).sum()),
        "cierres_sin_inversion": int(con_rubro.loc[~fin_mask, "C_falla"].sum()),
        "n_con_inversion": int(fin_mask.sum()),
        "cierres_con_inversion": int(con_rubro.loc[fin_mask, "C_falla"].sum()),
    }
    rub_all = con_rubro["category_code"].astype("category")
    res_has = M.ajustar_weibull_perfilado(
        con_rubro["T_meses"], con_rubro["C_falla"], rub_all.cat.codes.to_numpy(),
        fin_mask.to_numpy(float)[:, None], len(rub_all.cat.categories))
    se_has = float(np.sqrt(res_has["cov"][1, 1]))
    desc["hr_con_vs_sin_inversion"] = float(np.exp(res_has["beta"][0]))
    desc["hr_con_vs_sin_ic"] = [float(np.exp(res_has["beta"][0] - 1.96 * se_has)),
                                float(np.exp(res_has["beta"][0] + 1.96 * se_has))]

    # --- Modelo del simulador: solo empresas financiadas ------------------------------------
    f = con_rubro[fin_mask].copy()
    rubro_cat = f["category_code"].astype("category")
    rubros = list(rubro_cat.cat.categories)
    grupo = rubro_cat.cat.codes.to_numpy()
    t = f["T_meses"].to_numpy(float)
    c = f["C_falla"].to_numpy(float)
    x = M.x_desde_usd(f["raised_amount_usd"].to_numpy(float))

    ajuste = M.ajustar_weibull_perfilado(t, c, grupo, x[:, None], len(rubros))
    k, beta = ajuste["k"], float(ajuste["beta"][0])
    se_beta = float(np.sqrt(ajuste["cov"][1, 1]))
    print(f"  Weibull (financiadas): k = {k:.3f} | beta = {beta:.3f} (EE {se_beta:.3f}) "
          f"| HR por 10x capital = {np.exp(beta):.2f} | convergió: {ajuste['convergio']}")

    est = M.estimar_rubros(t, c, grupo, x, k, beta, len(rubros))
    n_por_rubro = np.bincount(grupo, minlength=len(rubros))
    rubros_json = {}
    for j, nombre in enumerate(rubros):
        rubros_json[nombre] = {
            "n_financiadas": int(n_por_rubro[j]), "cierres": int(est["cierres"][j]),
            "theta_cruda": float(est["theta_crudo"][j]), "theta_ajustada": float(est["theta_ajustada"][j]),
            "sd_posterior": float(est["sd_posterior"][j]), "B": float(est["B"][j])}
    print(f"  Bayes empírico: tau = {np.sqrt(est['tau2']):.3f} | I2 = {est['I2']:.1%} "
          f"| S(120) típica (1 MM) = {np.exp(-np.exp(est['mu'])):.3f}")

    # --- Validación: modelo vs Kaplan-Meier, por tramo de capital ---------------------------
    theta_i = est["theta_ajustada"][grupo]
    tramo = np.digitize(f["raised_amount_usd"].to_numpy(float), TRAMOS_USD[1:-1], right=True)
    km_tramos, validacion = [], []
    for b, etiqueta in enumerate(ETIQUETAS_TRAMOS):
        m = tramo == b
        u, S = km(t[m], c[m])
        grilla = np.arange(0, HORIZONTE_KM + 1)
        S_grilla = [km_en(u, S, g) for g in grilla]
        en_riesgo = [int((t[m] >= g).sum()) for g in grilla]
        km_tramos.append({"etiqueta": etiqueta, "n": int(m.sum()), "cierres": int(c[m].sum()),
                          "meses": grilla.tolist(), "S": [round(s, 5) for s in S_grilla],
                          "en_riesgo": en_riesgo})
        fila = {"tramo": etiqueta, "n": int(m.sum()), "cierres": int(c[m].sum())}
        for h in (60, 120):
            pred = float(np.mean(M.supervivencia(h, theta_i[m], beta, x[m], k)))
            fila[f"S{h}_modelo"] = round(pred, 4)
            fila[f"S{h}_km"] = round(km_en(u, S, h), 4)
        validacion.append(fila)
    u_all, S_all = km(t, c)
    for h in (60, 120):
        pred = float(np.mean(M.supervivencia(h, theta_i, beta, x, k)))
        print(f"  Validación S({h}) global financiadas: modelo {pred:.4f} vs KM {km_en(u_all, S_all, h):.4f}")

    # --- Robustez: capital levantado SOLO en los primeros 12 meses (landmark) ----------------
    lm = con_rubro[(con_rubro["T_meses"] > LANDMARK_MESES) & (con_rubro["raised_temprano_usd"] > 0)].copy()
    cat_lm = lm["category_code"].astype("category")
    res_lm = M.ajustar_weibull_perfilado(
        lm["T_meses"] - LANDMARK_MESES, lm["C_falla"], cat_lm.cat.codes.to_numpy(),
        M.x_desde_usd(lm["raised_temprano_usd"].to_numpy(float))[:, None], len(cat_lm.cat.categories))
    b_lm = float(res_lm["beta"][0])
    se_lm = float(np.sqrt(res_lm["cov"][1, 1]))
    robustez = {"landmark_meses": LANDMARK_MESES, "n": int(len(lm)), "cierres": int(lm["C_falla"].sum()),
                "beta": b_lm, "se_beta": se_lm, "hr_por_10x": float(np.exp(b_lm)),
                "hr_ic": [float(np.exp(b_lm - 1.96 * se_lm)), float(np.exp(b_lm + 1.96 * se_lm))]}
    print(f"  Robustez (landmark {LANDMARK_MESES} m): HR por 10x = {np.exp(b_lm):.2f} "
          f"[{robustez['hr_ic'][0]:.2f}, {robustez['hr_ic'][1]:.2f}]")

    salida = {
        "meta": {
            "fuente": "Crunchbase 2013 Snapshot (Kaggle, Justinas Baltrusaitis)",
            "fecha_corte": str(FECHA_CORTE.date()),
            "n_muestra_total": int(len(df)), "n_con_rubro": int(len(con_rubro)),
            "n_financiadas": int(len(f)), "cierres_financiadas": int(c.sum()),
            "horizonte_ref_meses": M.HORIZONTE_REF, "log10_ref": M.LOG10_REF,
            "monto_min_usd": float(f["raised_amount_usd"].quantile(0.01)),
            "monto_max_usd": float(f["raised_amount_usd"].quantile(0.995)),
        },
        "k": k, "beta": beta, "se_beta": se_beta, "hr_por_10x": float(np.exp(beta)),
        "cov_logk_beta": ajuste["cov"].tolist(),
        "mu": est["mu"], "tau2": est["tau2"], "I2": est["I2"], "Q": est["Q"], "p_Q": est["p_Q"],
        "rubros": rubros_json, "km_tramos": km_tramos, "validacion": validacion,
        "descriptivo_sin_vs_con": desc, "robustez_landmark": robustez,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo -> {args.salida}  ({args.salida.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
