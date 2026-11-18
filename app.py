"""
Calcula tu Supervivencia Empresarial — simulador interactivo para feria de análisis de datos.

Proyecto ICD 3152 Estadística Avanzada (PUCV). Datos: Crunchbase 2013 Snapshot (Kaggle).
Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from src import escenario as E
from src import graficos as G
from src.rubros import etiqueta, fmt_pct, fmt_usd

st.set_page_config(page_title="Calcula tu Supervivencia Empresarial", page_icon="🚀", layout="wide")

MONTOS = [25e3, 50e3, 75e3, 100e3, 150e3, 250e3, 500e3, 750e3, 1e6, 1.5e6, 2e6, 3e6, 5e6, 7.5e6,
          10e6, 15e6, 25e6, 50e6, 75e6, 100e6, 150e6, 250e6]


@st.cache_data
def cargar():
    return E.cargar_modelo()


def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def num(v: float, dec: int = 1) -> str:
    return f"{v:.{dec}f}".replace(".", ",")


def signo(v: float, dec: int = 1) -> str:
    """Con signo explícito ('+' o '-'): st.metric usa el signo para colorear el delta."""
    return f"{v:+.{dec}f}".replace(".", ",")


modelo = cargar()
meta = modelo["meta"]
rubros_ordenados = sorted(modelo["rubros"], key=lambda r: -modelo["rubros"][r]["n_financiadas"])

# ---------------------------------------------------------------------------------------------
st.title("🚀 Calcula tu Supervivencia Empresarial")
st.markdown(
    f"Configura una startup ficticia y mira cómo **el capital de riesgo desplaza el riesgo de cierre hacia "
    f"adelante en el tiempo**. Modelo estadístico (Weibull + Bayes empírico) ajustado con "
    f"**{miles(meta['n_financiadas'])} startups financiadas** de Crunchbase."
)

col_ctrl, col_res = st.columns([1, 2.3], gap="large")

# ----------------------------- Controles ------------------------------------------------------
with col_ctrl:
    st.subheader("1 · Arma tu startup")
    rubro = st.selectbox(
        "Rubro", rubros_ordenados, index=rubros_ordenados.index("software"),
        format_func=lambda r: f"{etiqueta(r)}  ({miles(modelo['rubros'][r]['n_financiadas'])} startups)")
    monto = st.select_slider("Capital levantado (`raised_amount_usd`)", options=MONTOS, value=1e6,
                             format_func=fmt_usd)
    horizonte = st.radio("Horizonte de los gráficos", [60, 120], index=1, horizontal=True,
                         format_func=lambda h: f"{h // 12} años")
    with st.expander("Opciones avanzadas"):
        mostrar_obs = st.checkbox("Superponer datos observados (Kaplan-Meier)", value=True)
        conservador = st.checkbox(
            "Escenario conservador del efecto del capital", value=False,
            help="Usa la pendiente estimada solo con el capital levantado en los primeros 12 meses "
                 "(evita el sesgo de 'tiempo inmortal'). El efecto del capital es menor.")

    if monto > 1e8:
        st.warning("Más de US$ 100 millones: hay solo 514 startups en los datos. Interpreta con cautela.")

    res = E.calcular(modelo, rubro, monto, conservador)
    st.markdown("##### 100 startups como la tuya")
    S_h = res["tu"]["S60"] if horizonte == 60 else res["tu"]["S120"]
    st.pyplot(G.fig_waffle(S_h, f"{horizonte // 12} años"))
    cerradas = (1 - S_h) * 100
    st.caption(f"≈ **{num(cerradas)}** de cada 100 habrían cerrado a los {horizonte // 12} años (rojo).")

# ----------------------------- Resultados -----------------------------------------------------
with col_res:
    st.subheader("2 · Lo que dicen los datos")
    tu, ref = res["tu"], res["ref"]
    r_info = modelo["rubros"][rubro]
    ref_txt = fmt_usd(E.MONTO_REFERENCIA)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Abiertas a 5 años", fmt_pct(tu["S60"]),
              f"{signo(100 * (tu['S60'] - ref['S60']))} pts vs {ref_txt}")
    c2.metric("Abiertas a 10 años", fmt_pct(tu["S120"]),
              f"{signo(100 * (tu['S120'] - ref['S120']))} pts vs {ref_txt}")
    ayuda_t = (f"Mes en que habría cerrado el 5 % de las startups con este perfil. "
               f"Con {ref_txt}: " + ("más de 10 años." if ref["t_cierre_fuera"]
                                     else f"{num(ref['t_cierre'], 0)} meses."))
    if tu["t_cierre_fuera"]:
        c3.metric("Mes en que habrá cerrado el 5 %", "más de 10 años", help=ayuda_t)
    elif ref["t_cierre_fuera"]:
        c3.metric("Mes en que habrá cerrado el 5 %", f"{num(tu['t_cierre'], 0)} meses", help=ayuda_t)
    else:
        c3.metric("Mes en que habrá cerrado el 5 %", f"{num(tu['t_cierre'], 0)} meses",
                  f"{signo(tu['t_cierre'] - ref['t_cierre'], 0)} meses vs {ref_txt}", help=ayuda_t)
    c4.metric("Riesgo de cierre vs " + ref_txt, f"{num(res['hr_vs_ref'], 2)}×",
              f"{signo(100 * (res['hr_vs_ref'] - 1), 0)} %", delta_color="inverse")

    st.success(
        f"**Con {fmt_usd(monto)} en {etiqueta(rubro)}**, estimamos que a los 5 años habrían cerrado "
        f"≈ **{num(100 * (1 - tu['S60']))} de cada 100** startups parecidas (IC 95 %: "
        f"{num(100 * (1 - tu['S60_ic'][1]))}–{num(100 * (1 - tu['S60_ic'][0]))}), frente a "
        f"≈ {num(100 * (1 - ref['S60']))} con {ref_txt}.")

    if r_info["cierres"] == 0:
        st.warning(f"En este rubro no hay cierres registrados entre sus {miles(r_info['n_financiadas'])} startups "
                   "financiadas: la estimación se apoya casi por completo en el promedio de todos los rubros "
                   "(Bayes empírico), por eso el intervalo es ancho.")
    elif r_info["cierres"] < 10:
        st.info(f"Solo {r_info['cierres']} cierres observados en este rubro: la estimación se acerca "
                f"{num(100 * r_info['B'], 0)} % al promedio de todos los rubros (contracción de Bayes empírico).")
    else:
        st.caption(f"Rubro con {miles(r_info['n_financiadas'])} startups financiadas y {r_info['cierres']} cierres "
                   f"· contracción hacia el promedio: {num(100 * r_info['B'], 0)} %.")

    t1, t2, t3, t4, t5 = st.tabs(["📈 Supervivencia", "⚠️ Riesgo", "💰 Efecto del capital",
                                  "🏭 Comparar rubros", "🔬 Cómo se hizo y límites"])
    with t1:
        st.pyplot(G.fig_supervivencia(modelo, rubro, monto, horizonte, conservador, mostrar_obs))
        st.caption("La zona verde entre las curvas es la supervivencia que se 'gana' con el capital: "
                   "la curva de tu startup se desplaza hacia arriba y hacia la derecha.")
    with t2:
        st.pyplot(G.fig_riesgo(modelo, rubro, monto, horizonte, conservador))
        st.caption(f"Entre startups financiadas el riesgo mensual de cierre **sube levemente con la edad** "
                   f"(forma Weibull k = {num(modelo['k'], 2)}). Cada 10× más capital multiplica el riesgo por "
                   f"{num(float(np.exp(res['beta'])), 2)} (escenario {'conservador' if conservador else 'base'}).")
    with t3:
        st.pyplot(G.fig_efecto_capital(modelo, rubro, monto, conservador))
        rob = modelo["robustez_landmark"]
        st.caption(f"Modelo base: cada 10× más capital → riesgo × {num(modelo['hr_por_10x'], 2)} "
                   f"(IC 95 %: {num(float(np.exp(modelo['beta'] - 1.96 * modelo['se_beta'])), 2)}–"
                   f"{num(float(np.exp(modelo['beta'] + 1.96 * modelo['se_beta'])), 2)}). "
                   f"Escenario conservador (solo capital de los primeros {rob['landmark_meses']} meses): "
                   f"× {num(rob['hr_por_10x'], 2)} (IC 95 %: {num(rob['hr_ic'][0], 2)}–{num(rob['hr_ic'][1], 2)}).")
    with t4:
        st.pyplot(G.fig_rubros(modelo, rubro, monto, horizonte, conservador))
        st.caption("Los rubros con pocos cierres (entre paréntesis) están fuertemente contraídos hacia el promedio; "
                   "sus diferencias con otros rubros deben leerse con cautela.")
    with t5:
        d = modelo["descriptivo_sin_vs_con"]
        rob = modelo["robustez_landmark"]
        st.markdown(f"""
**Datos.** Crunchbase 2013 Snapshot (Kaggle). Muestra del proyecto: {miles(meta['n_muestra_total'])} empresas con fecha de
fundación; el simulador usa las **{miles(meta['n_financiadas'])} con financiamiento** y rubro conocido
({meta['cierres_financiadas']} cierres). El tiempo observado se mide hasta el cierre o hasta el corte (2013-12-31); las
empresas vivas, adquiridas o con IPO son **censuras por la derecha**.

**Modelo.** Weibull con riesgo proporcional: `H(t) = exp(θ_rubro + β·x)·(t/120)^k`, con
`x = log10(capital) − 6`. La forma `k` y la pendiente `β` se estiman por máxima verosimilitud (rubros como efectos
fijos, verosimilitud perfilada). El nivel de cada rubro `θ_rubro` se estima por **Bayes empírico (DerSimonian-Laird)**,
igual que el Método 3 del proyecto: los rubros con pocos cierres se acercan al promedio. Las bandas son intervalos
aproximados al 95 % (simulando k, β y θ).

**Resultados clave.** k = {num(modelo['k'], 3)}; cada 10× más capital → riesgo × {num(modelo['hr_por_10x'], 2)};
heterogeneidad entre rubros I² = {num(100 * modelo['I2'], 0)} %.
""")
        st.markdown("##### Validación: modelo vs. Kaplan-Meier por tramo de capital")
        val = pd.DataFrame(modelo["validacion"]).rename(columns={
            "tramo": "Tramo de capital", "n": "Startups", "cierres": "Cierres",
            "S60_modelo": "S(60) modelo", "S60_km": "S(60) observado",
            "S120_modelo": "S(120) modelo", "S120_km": "S(120) observado"})
        st.dataframe(val, hide_index=True)
        st.markdown(f"""
##### Límites que conviene decir en voz alta
1. **Asociación, no causalidad.** Las startups más sólidas probablemente levantan más capital; el modelo mide
   diferencias entre empresas, no el efecto de "inyectar" dinero a una empresa dada.
2. **Sesgo de tiempo inmortal.** `raised_amount_usd` es el total acumulado en toda la vida de la empresa: sobrevivir más
   da más tiempo para levantar capital. Como chequeo, usando solo el capital de los primeros {rob['landmark_meses']} meses
   ({miles(rob['n'])} startups) el efecto se reduce a × {num(rob['hr_por_10x'], 2)} por 10× (escenario conservador),
   pero la dirección se mantiene.
3. **Las startups sin inversión no son comparables.** Entre las {miles(d['n_sin_inversion'])} sin inversión se registran
   {d['cierres_sin_inversion']} cierres (≈ {num(100 * d['cierres_sin_inversion'] / d['n_sin_inversion'], 1)} %), y entre las
   {miles(d['n_con_inversion'])} con inversión, {miles(d['cierres_con_inversion'])} (≈
   {num(100 * d['cierres_con_inversion'] / d['n_con_inversion'], 1)} %): ajustando por rubro, tener inversión se asocia con
   un riesgo registrado ≈ {num(d['hr_con_vs_sin_inversion'], 1)}× mayor. Es contraintuitivo y probablemente refleja
   que Crunchbase registra mejor los cierres de empresas con financiamiento (sub-registro de cierres de
   empresas pequeñas); no podemos comprobarlo con estos datos. Por eso el simulador **solo cubre startups financiadas**.
4. **Supervivencia registrada, foto de 2013.** Las adquiridas o con IPO se tratan como "sin cierre". Los cierres de
   empresas pequeñas suelen no anotarse, así que la supervivencia real podría ser menor.
5. **Forma del riesgo.** Entre financiadas k ≈ {num(modelo['k'], 2)} (riesgo creciente); en la muestra completa del
   Método 1 k ≈ 0,81 (decreciente), porque esa muestra mezcla las no financiadas.
6. **Ajuste imperfecto en los extremos.** La forma log-lineal subestima algo la supervivencia en el tramo de menos de
   US$ 100 mil y a 10 años en general (ver tabla). Más allá de la zona sombreada hay pocos datos observados.
""")

st.divider()
st.caption("Proyecto ICD 3152 Estadística Avanzada · PUCV · Datos: Crunchbase 2013 Snapshot (Kaggle, J. Baltrušaitis). "
           "Herramienta educativa: no es asesoría de inversión.")
