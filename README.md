# 🧠 Hábitos digitales y bienestar en adolescentes · EDA con Streamlit

**Caso de Estudio N°4** · Especialización en Python for Analytics (DMC Institute) · 2026
**Autor:** Rodrigo Enrique Huarcaya Galarza



## Descripción del proyecto

Aplicación interactiva en Streamlit para explorar el dataset `Teen_Mental_Health_Dataset.csv`
(1.200 adolescentes de 13 a 19 años, 13 variables). Analiza cómo se relacionan el uso de redes
sociales, el sueño, la actividad física y la interacción social con las escalas de estrés,
ansiedad y dependencia, y con `depression_label`, una etiqueta binaria propia del dataset.
No se construyen modelos predictivos: el foco es describir, comparar grupos y comunicar con
cautela.

La app tiene tres módulos en el sidebar:

1. **Home**: título, objetivo, autor, dataset y tecnologías (sin análisis).
2. **Carga del dataset**: `st.file_uploader`, validación de columnas, `head()` y dimensiones.
   Ningún análisis corre si no hay archivo cargado.
3. **EDA**: 10 ítems en tabs (info, clasificación, descriptivas, faltantes, distribuciones,
   categóricas, numérico vs categórico, categórico vs categórico, análisis interactivo y
   hallazgos con 5 conclusiones).

Widgets usados: `sidebar`, `tabs`, `columns`, `selectbox`, `multiselect`, `slider`, `checkbox`.
La lógica está encapsulada en la clase `DataAnalyzer` (POO) y la función personalizada
`clasificar_variables()`.

## Capturas de la app

Agrega aquí tus capturas (carpeta `screenshots/`):

![Home](screenshots/home.png)
![EDA](screenshots/eda.png)

## Instrucciones de ejecución

```bash
git clone <URL-DE-TU-REPOSITORIO>
cd teen-mental-health-eda
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Luego, en el módulo **Carga del dataset**, sube `Teen_Mental_Health_Dataset.csv` (o marca la
casilla para usar la copia incluida en el repositorio).

## Variables principales

| Variable | Descripción |
|---|---|
| `age`, `gender` | Edad (13-19) y género registrado |
| `daily_social_media_hours` | Horas diarias de redes sociales |
| `platform_usage` | Instagram, TikTok o Both |
| `sleep_hours` | Horas de sueño por día |
| `screen_time_before_sleep` | Horas de pantalla antes de dormir |
| `academic_performance`, `physical_activity` | Rendimiento académico y horas de actividad física |
| `social_interaction_level` | low, medium o high |
| `stress_level`, `anxiety_level`, `addiction_level` | Escalas de 1 a 10 |
| `depression_label` | Etiqueta binaria del dataset (0 = ausencia, 1 = presencia) |

## Hallazgos resumidos

- Solo 31 de 1.200 registros (2,6 %) tienen `depression_label = 1`: toda comparación es frágil.
- Ese grupo declara más horas de redes (6,7 vs 4,5 h), menos sueño (4,8 vs 6,5 h) y mayores
  puntajes de estrés y ansiedad.
- Plataforma, género e interacción social casi no diferencian entre grupos.
- Son asociaciones descriptivas, no causalidad.

## Links relevantes

- Repositorio: _pega aquí el link de GitHub_
- App desplegada: _pega aquí el link de Streamlit Community Cloud_
- [Streamlit](https://streamlit.io) · [Pandas](https://pandas.pydata.org) · [Seaborn](https://seaborn.pydata.org)
