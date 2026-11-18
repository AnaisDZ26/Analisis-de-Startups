# 🚀 Calcula tu Supervivencia Empresarial

Simulador interactivo para feria de análisis de datos. Proyecto ICD 3152 Estadística Avanzada (PUCV).
El visitante elige **rubro** y **capital levantado** (`raised_amount_usd`) y ve en tiempo real la curva de
supervivencia, la función de riesgo y cuántas de cada 100 startups parecidas habrían cerrado.

## Inicio rápido (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`data/modelo.json` (29 KB) ya viene generado. **No necesitas los CSV de Kaggle para correr la app.**

## Regenerar el modelo (solo si cambias el análisis)

```bash
python scripts/preparar_modelo.py --datos ruta/a/dataset.zip     # o una carpeta con los CSV
python tests/test_modelo.py                                       # 7 pruebas de sanidad
```

## Estructura

```
app.py                      interfaz Streamlit (controles, métricas, pestañas)
src/modelo.py               Weibull perfilado + Bayes empírico (DerSimonian-Laird) + curvas + bandas
src/escenario.py            cifras de un escenario (rubro + monto) para la interfaz
src/graficos.py             gráficos matplotlib (pensados para tablet: letra grande)
src/rubros.py               nombres en español y formato de montos
scripts/preparar_modelo.py  replica el pipeline del notebook y escribe data/modelo.json
data/modelo.json            parámetros del modelo (lo único que lee la app)
tests/test_modelo.py        pruebas
```

## Modelo

Solo startups **con financiamiento** (21.318; 1.287 cierres), `H(t) = exp(θ_rubro + β·x)·(t/120)^k`,
`x = log10(capital) − 6`. `k` y `β` por máxima verosimilitud con rubros como efectos fijos (verosimilitud
perfilada); `θ_rubro` por Bayes empírico, como el Método 3 del proyecto. Bandas al 95 % simulando k, β y θ.

Resultados: k = 1,26; cada 10× más capital → riesgo × 0,48 (escenario conservador, solo capital de los primeros
12 meses: × 0,61).

## Despliegue

**Recomendación para la feria: correr local como principal y dejar la nube como respaldo / para el QR.**

| | Local (tu notebook) | Streamlit Community Cloud |
|---|---|---|
| Internet en el stand | No lo necesita (una vez instaladas las dependencias) | Imprescindible |
| Riesgo | Cero dependencia de terceros | La app se duerme tras 12 h sin tráfico y hay que despertarla |
| Visitantes con su celular | Solo en la misma red (`http://IP-de-tu-notebook:8501`) | Cualquiera, con un QR |

Modo kiosco en el notebook/tablet: `streamlit run app.py` y abrir `chrome --kiosk http://localhost:8501`.
Para la nube: sube la carpeta a un repositorio de GitHub (sin los CSV: ya están en `.gitignore`), entra a
share.streamlit.io, elige el repo y `app.py`. **Despierta la app la noche anterior y de nuevo en la mañana.**

## Guion de 60 segundos para el stand

1. "Elige un rubro y cuánto capital levantó tu startup."
2. "La curva verde es tu startup; la gris, la misma con solo US$ 100 mil. El área verde es la supervivencia que gana el capital."
3. "Mira los 100 puntos: los rojos son las startups parecidas que habrían cerrado."
4. "¿Es causal? No: es una asociación, y por eso mostramos el escenario conservador."
# Analisis-de-Startups
