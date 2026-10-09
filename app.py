"""
EDA interactivo: hábitos digitales y bienestar en adolescentes
Caso de Estudio N°4 - Especialización en Python for Analytics (DMC Institute)

Ejecutar con:  streamlit run app.py
Nota: el análisis es educativo y exploratorio. No constituye diagnóstico clínico.
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Variables globales (edita los datos del autor)
# ---------------------------------------------------------------------------
AUTOR_NOMBRE = "Rodrigo Enrique Huarcaya Galarza"
AUTOR_CURSO = "Especialización en Python for Analytics - DMC Institute"
AUTOR_ANIO = 2026
RUTA_LOCAL = "Teen_Mental_Health_Dataset.csv"

COLUMNAS_ESPERADAS = [
    "age", "gender", "daily_social_media_hours", "platform_usage", "sleep_hours",
    "screen_time_before_sleep", "academic_performance", "physical_activity",
    "social_interaction_level", "stress_level", "anxiety_level", "addiction_level",
    "depression_label",
]
ORDEN_INTERACCION = ["low", "medium", "high"]
VARS_BIENESTAR = ["stress_level", "anxiety_level", "addiction_level", "sleep_hours"]
VARS_HABITOS = ["daily_social_media_hours", "screen_time_before_sleep",
                "physical_activity", "academic_performance"]

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Función personalizada (Ítem 2)
# ---------------------------------------------------------------------------
def clasificar_variables(df):
    """Separa las columnas en numéricas y categóricas.

    Una columna numérica con solo 2 valores distintos (0/1) se considera
    categórica, porque es una etiqueta y no una medida.
    """
    numericas = []
    categoricas = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2:
            numericas.append(col)
        else:
            categoricas.append(col)
    return numericas, categoricas


# ---------------------------------------------------------------------------
# Clase principal (POO)
# ---------------------------------------------------------------------------
class DataAnalyzer:
    """Guarda el DataFrame y agrupa estadística, filtros y gráficos."""

    def __init__(self, df):
        self.df = df
        self.target = "depression_label"
        self.numericas, self.categoricas = clasificar_variables(df)

    # ----- validación e información general -----
    def columnas_faltantes(self):
        faltantes = []
        for col in COLUMNAS_ESPERADAS:
            if col not in self.df.columns:
                faltantes.append(col)
        return faltantes

    def info_texto(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def resumen_tipos(self):
        tabla = pd.DataFrame()
        tabla["tipo"] = self.df.dtypes.astype(str)
        tabla["nulos"] = self.df.isnull().sum()
        tabla["únicos"] = self.df.nunique()
        return tabla

    # ----- estadística descriptiva -----
    def tabla_central(self):
        d = self.df[self.numericas]
        tabla = pd.DataFrame()
        tabla["media"] = d.mean()
        tabla["mediana"] = d.median()
        tabla["moda"] = d.mode().iloc[0]
        tabla["desv. est."] = d.std()
        tabla["asimetría"] = d.skew()
        return tabla.round(2)

    def outliers_iqr(self):
        filas = []
        for col in self.numericas:
            q1 = np.percentile(self.df[col], 25)
            q3 = np.percentile(self.df[col], 75)
            iqr = q3 - q1
            inferior = q1 - 1.5 * iqr
            superior = q3 + 1.5 * iqr
            n = ((self.df[col] < inferior) | (self.df[col] > superior)).sum()
            filas.append([col, round(inferior, 2), round(superior, 2), n])
        return pd.DataFrame(filas, columns=["variable", "límite inf.", "límite sup.", "outliers"])

    def faltantes(self):
        tabla = pd.DataFrame()
        tabla["nulos"] = self.df.isnull().sum()
        tabla["% nulos"] = (tabla["nulos"] / len(self.df) * 100).round(2)
        return tabla

    # ----- filtros y comparación de grupos -----
    def filtrar(self, edad, generos, plataformas, interaccion):
        d = self.df
        d = d[(d["age"] >= edad[0]) & (d["age"] <= edad[1])]
        d = d[d["gender"].isin(generos)]
        d = d[d["platform_usage"].isin(plataformas)]
        d = d[d["social_interaction_level"].isin(interaccion)]
        return d

    def comparar_grupos(self, variables, df):
        medias = df.groupby(self.target)[variables].mean().round(2)
        medianas = df.groupby(self.target)[variables].median().round(2)
        return medias, medianas

    def tasa_por_categoria(self, col, df):
        casos = df.groupby(col)[self.target].sum()
        total = df.groupby(col)[self.target].count()
        tabla = pd.DataFrame()
        tabla["casos (etiqueta=1)"] = casos
        tabla["total"] = total
        tabla["tasa (%)"] = (casos / total * 100).round(2)
        return tabla

    # ----- gráficos -----
    def hist(self, col, bins=20, kde=True):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.histplot(data=self.df, x=col, bins=bins, kde=kde, ax=ax)
        ax.axvline(self.df[col].mean(), color="black", linestyle="--", label="media")
        ax.axvline(self.df[col].median(), color="gray", linestyle=":", label="mediana")
        ax.legend()
        ax.set_title(col)
        fig.tight_layout()
        return fig

    def conteo(self, col, orden=None):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        sns.countplot(data=self.df, x=col, order=orden, ax=ax, color="#4C78A8")
        ax.set_title("Conteo de " + col)
        fig.tight_layout()
        return fig

    def boxplot_grupo(self, col):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        sns.boxplot(data=self.df, x=self.target, y=col, ax=ax)
        ax.set_title(col + " según " + self.target)
        fig.tight_layout()
        return fig

    def barras_tasa(self, col, orden=None):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        tasas = self.df.groupby(col)[self.target].mean() * 100
        tasas = tasas.reset_index()
        sns.barplot(data=tasas, x=col, y=self.target, order=orden, ax=ax, color="#E4572E")
        ax.set_ylabel("% con etiqueta = 1")
        ax.set_title("Proporción de etiqueta = 1 por " + col)
        fig.tight_layout()
        return fig

    def conteo_por_etiqueta(self, col, orden=None):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        sns.countplot(data=self.df, x=col, hue=self.target, order=orden, ax=ax)
        ax.set_title(col + " vs " + self.target)
        fig.tight_layout()
        return fig

    def heatmap_corr(self, columnas, df):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(df[columnas].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title("Correlación (asociación, no causalidad)")
        fig.tight_layout()
        return fig

    def dispersion(self, x, y, df, colorear):
        fig, ax = plt.subplots(figsize=(6, 4))
        if colorear:
            sns.scatterplot(data=df, x=x, y=y, hue=self.target, alpha=0.6, ax=ax)
        else:
            sns.scatterplot(data=df, x=x, y=y, alpha=0.6, ax=ax)
        ax.set_title(y + " vs " + x)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Módulo 1: Home
# ---------------------------------------------------------------------------
def pagina_home():
    st.title("🧠 Hábitos digitales y bienestar en adolescentes")
    st.caption("Análisis Exploratorio de Datos (EDA) · Caso de Estudio N°4")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("🎯 Objetivo del análisis")
        st.write("Explorar patrones entre el uso de redes sociales, el descanso, la actividad "
                 "física, la interacción social y las escalas de estrés, ansiedad y dependencia "
                 "en adolescentes de 13 a 19 años, para apoyar decisiones de prevención y "
                 "educación digital. No se construyen modelos predictivos.")
        st.subheader("🗂️ Sobre el dataset")
        st.write("`Teen_Mental_Health_Dataset.csv` tiene 1.200 registros y 13 variables: edad, "
                 "género, horas de redes sociales, plataforma, horas de sueño, pantalla antes de "
                 "dormir, rendimiento académico, actividad física, interacción social, estrés, "
                 "ansiedad y dependencia (escalas 1-10) y `depression_label`, una etiqueta "
                 "binaria propia del dataset.")
    with col2:
        st.subheader("👤 Autor")
        st.write(f"**Nombre:** {AUTOR_NOMBRE}")
        st.write(f"**Curso:** {AUTOR_CURSO}")
        st.write(f"**Año:** {AUTOR_ANIO}")
        st.subheader("🛠️ Tecnologías")
        st.write("Python · Pandas · Matplotlib · Seaborn · Streamlit · GitHub")


# ---------------------------------------------------------------------------
# Módulo 2: Carga del dataset
# ---------------------------------------------------------------------------
def pagina_carga():
    st.title("📥 Carga del dataset")
    archivo = st.file_uploader("Sube el archivo CSV", type=["csv"])
    usar_local = False
    if os.path.exists(RUTA_LOCAL):
        usar_local = st.checkbox("Usar el dataset incluido en el repositorio")

    if archivo is not None:
        df = pd.read_csv(archivo)
    elif usar_local:
        df = pd.read_csv(RUTA_LOCAL)
    else:
        if "df" in st.session_state:
            del st.session_state["df"]
        st.warning("⚠️ No se ha cargado ningún archivo. No se ejecutará ningún análisis.")
        return

    faltantes = DataAnalyzer(df).columnas_faltantes()
    if len(faltantes) > 0:
        if "df" in st.session_state:
            del st.session_state["df"]
        st.error(f"El archivo no tiene el formato esperado. Columnas faltantes: {faltantes}")
        return

    st.session_state["df"] = df
    st.success(f"✅ Archivo cargado correctamente ({df.shape[0]} filas).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Nulos totales", int(df.isnull().sum().sum()))
    st.subheader("Vista previa (head)")
    n = st.slider("Filas a mostrar", 5, 50, 10)
    st.dataframe(df.head(n))


# ---------------------------------------------------------------------------
# Módulo 3: EDA (10 ítems)
# ---------------------------------------------------------------------------
def pagina_eda():
    st.title("📊 Análisis Exploratorio de Datos")
    if "df" not in st.session_state:
        st.warning("⚠️ Primero carga el dataset en el módulo **Carga del dataset**.")
        return

    an = DataAnalyzer(st.session_state["df"])
    df = an.df
    t = an.target

    tabs = st.tabs(["1 Info", "2 Variables", "3 Descriptivas", "4 Faltantes", "5 Distribución",
                    "6 Categóricas", "7 Num vs Cat", "8 Cat vs Cat", "9 Interactivo",
                    "10 Hallazgos"])

    # ---- Ítem 1 ----
    with tabs[0]:
        st.header("Ítem 1 · Información general")
        st.write("Estructura del dataset: tipos de dato, nulos y duplicados.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("`.info()`")
            st.code(an.info_texto())
        with c2:
            st.subheader("Tipos, nulos y únicos")
            st.dataframe(an.resumen_tipos())
            st.metric("Nulos totales", int(df.isnull().sum().sum()))
            st.metric("Filas duplicadas", int(df.duplicated().sum()))
        st.caption("Sin nulos ni duplicados: no hace falta limpiar, solo validar tipos.")

    # ---- Ítem 2 ----
    with tabs[1]:
        st.header("Ítem 2 · Clasificación de variables")
        st.write("La función `clasificar_variables()` separa numéricas de categóricas. "
                 "`depression_label` (0/1) se trata como categórica porque es una etiqueta.")
        c1, c2 = st.columns(2)
        c1.metric("Numéricas", len(an.numericas))
        c1.write(an.numericas)
        c2.metric("Categóricas", len(an.categoricas))
        c2.write(an.categoricas)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Numéricas", "Categóricas"], [len(an.numericas), len(an.categoricas)])
        ax.set_ylabel("Cantidad")
        st.pyplot(fig)

    # ---- Ítem 3 ----
    with tabs[2]:
        st.header("Ítem 3 · Estadísticas descriptivas")
        st.subheader("`.describe()`")
        st.dataframe(df[an.numericas].describe().round(2))
        st.subheader("Media, mediana, moda y dispersión")
        st.dataframe(an.tabla_central())
        st.subheader("Valores extremos (regla del rango intercuartil)")
        st.dataframe(an.outliers_iqr())
        st.write("**Lectura:** la media y la mediana son casi iguales en todas las variables, "
                 "así que las distribuciones son simétricas. Las escalas 1-10 tienen desviación "
                 "alta (≈2,9) porque los valores se reparten por todo el rango.")

    # ---- Ítem 4 ----
    with tabs[3]:
        st.header("Ítem 4 · Valores faltantes")
        c1, c2 = st.columns([2, 3])
        c1.dataframe(an.faltantes())
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
        ax.set_title("Mapa de nulos")
        c2.pyplot(fig)
        if df.isnull().sum().sum() == 0:
            st.success("No hay valores faltantes: se conservan todos los registros.")
        else:
            st.warning("Hay valores faltantes: documenta el criterio antes de imputar o eliminar.")

    # ---- Ítem 5 ----
    with tabs[4]:
        st.header("Ítem 5 · Distribución de variables numéricas")
        c1, c2, c3 = st.columns(3)
        col = c1.selectbox("Variable", an.numericas)
        bins = c2.slider("Bins", 5, 50, 20)
        kde = c3.checkbox("Mostrar curva KDE", value=True)
        st.pyplot(an.hist(col, bins, kde))
        st.caption(f"Media {df[col].mean():.2f} · mediana {df[col].median():.2f} · "
                   f"asimetría {df[col].skew():.2f}")
        st.subheader("Estrés, ansiedad y dependencia")
        c4, c5, c6 = st.columns(3)
        c4.pyplot(an.hist("stress_level", 10, False))
        c5.pyplot(an.hist("anxiety_level", 10, False))
        c6.pyplot(an.hist("addiction_level", 10, False))
        st.write("Las tres escalas (1-10) son amplias y casi planas. "
                 "**No son un diagnóstico clínico**, solo puntajes del dataset.")

    # ---- Ítem 6 ----
    with tabs[5]:
        st.header("Ítem 6 · Variables categóricas")
        cat = st.selectbox("Variable categórica", an.categoricas)
        orden = None
        if cat == "social_interaction_level":
            orden = ORDEN_INTERACCION
        c1, c2 = st.columns([3, 2])
        c1.pyplot(an.conteo(cat, orden))
        tabla = pd.DataFrame()
        tabla["conteo"] = df[cat].value_counts()
        tabla["proporción (%)"] = (df[cat].value_counts(normalize=True) * 100).round(1)
        c2.dataframe(tabla)
        st.caption(f"Moda de {cat}: {df[cat].mode()[0]}")
        if cat == t:
            st.warning(f"Etiqueta muy desbalanceada: solo {int(df[t].sum())} registros con 1.")

    # ---- Ítem 7 ----
    with tabs[6]:
        st.header("Ítem 7 · Numérico vs categórico (según la etiqueta)")
        var = st.selectbox("Variable numérica", an.numericas, index=1)
        c1, c2 = st.columns(2)
        c1.pyplot(an.boxplot_grupo(var))
        medias, medianas = an.comparar_grupos([var], df)
        c2.write("Media por grupo")
        c2.dataframe(medias)
        c2.write("Mediana por grupo")
        c2.dataframe(medianas)
        st.subheader("Variables clave")
        clave = ["daily_social_media_hours", "sleep_hours", "academic_performance",
                 "physical_activity"]
        medias, medianas = an.comparar_grupos(clave, df)
        st.dataframe(medias)
        st.caption(f"El grupo 1 tiene solo {int(df[t].sum())} registros: las diferencias son "
                   "descriptivas y no prueban causalidad.")

    # ---- Ítem 8 ----
    with tabs[7]:
        st.header("Ítem 8 · Categórico vs categórico")
        cat = st.selectbox("Variable", ["platform_usage", "social_interaction_level", "gender"])
        orden = None
        if cat == "social_interaction_level":
            orden = ORDEN_INTERACCION
        c1, c2 = st.columns(2)
        c1.pyplot(an.barras_tasa(cat, orden))
        c2.pyplot(an.conteo_por_etiqueta(cat, orden))
        st.dataframe(an.tasa_por_categoria(cat, df))
        st.subheader("Género vs plataforma de uso")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="platform_usage", hue="gender", ax=ax)
        st.pyplot(fig)
        st.caption("Con tan pocos casos por categoría, diferencias de 1-2 puntos pueden ser azar.")

    # ---- Ítem 9 ----
    with tabs[8]:
        st.header("Ítem 9 · Análisis con parámetros seleccionados")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filtros (Ítem 9)")
        edad = st.sidebar.slider("Rango de edad", 13, 19, (13, 19))
        generos = st.sidebar.multiselect("Género", ["female", "male"], default=["female", "male"])
        plataformas = st.sidebar.multiselect("Plataforma", ["Instagram", "TikTok", "Both"],
                                             default=["Instagram", "TikTok", "Both"])
        interaccion = st.sidebar.multiselect("Interacción social", ORDEN_INTERACCION,
                                             default=ORDEN_INTERACCION)
        colorear = st.sidebar.checkbox("Colorear por etiqueta", value=True)

        d = an.filtrar(edad, generos, plataformas, interaccion)
        if len(d) == 0:
            st.error("Los filtros no devuelven registros. Amplía la selección en la barra lateral.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Registros filtrados", len(d))
            c2.metric("Etiqueta = 1", int(d[t].sum()))
            c3.metric("% etiqueta = 1", f"{d[t].mean() * 100:.1f}%")
            a, b = st.columns(2)
            bienestar = a.selectbox("Variable de bienestar", VARS_BIENESTAR)
            habito = b.selectbox("Variable de hábitos digitales", VARS_HABITOS)
            c4, c5 = st.columns(2)
            c4.pyplot(an.dispersion(habito, bienestar, d, colorear))
            c5.pyplot(an.heatmap_corr([habito, bienestar, "sleep_hours", t], d))
            st.write(f"**Correlación entre {habito} y {bienestar}:** "
                     f"{d[habito].corr(d[bienestar]):.2f}")
            medias, medianas = an.comparar_grupos([habito, bienestar], d)
            st.dataframe(medias)

    # ---- Ítem 10 ----
    with tabs[9]:
        mostrar_hallazgos(an)


def mostrar_hallazgos(an):
    df = an.df
    t = an.target
    st.header("Ítem 10 · Hallazgos clave")

    n1 = int(df[t].sum())
    if n1 == 0:
        st.warning("No hay registros con etiqueta 1; no se pueden comparar grupos.")
        return
    porcentaje1 = n1 / len(df) * 100

    medias = df.groupby(t)[an.numericas].mean()
    redes1 = medias.loc[1, "daily_social_media_hours"]
    redes0 = medias.loc[0, "daily_social_media_hours"]
    sueno1 = medias.loc[1, "sleep_hours"]
    sueno0 = medias.loc[0, "sleep_hours"]
    estres1 = medias.loc[1, "stress_level"]
    estres0 = medias.loc[0, "stress_level"]
    ansiedad1 = medias.loc[1, "anxiety_level"]
    ansiedad0 = medias.loc[0, "anxiety_level"]

    uso_alto = df[df["daily_social_media_hours"] >= 6]
    uso_bajo = df[df["daily_social_media_hours"] < 6]
    tasa_alta = uso_alto[t].mean() * 100
    tasa_baja = uso_bajo[t].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", len(df))
    c2.metric("Etiqueta = 1", f"{n1} ({porcentaje1:.1f}%)")
    c3.metric("Redes h/día (1 vs 0)", f"{redes1:.1f} vs {redes0:.1f}")
    c4.metric("Sueño h (1 vs 0)", f"{sueno1:.1f} vs {sueno0:.1f}")

    # Gráfico resumen: diferencia entre grupos medida en desviaciones estándar
    diferencias = (medias.loc[1] - medias.loc[0]) / df[an.numericas].std()
    diferencias = diferencias.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(diferencias.index, diferencias.values, color="#4C78A8")
    ax.axvline(0, color="black")
    ax.set_xlabel("Diferencia grupo 1 − grupo 0 (en desviaciones estándar)")
    ax.set_title("¿En qué variables se separan más los dos grupos?")
    fig.tight_layout()
    st.pyplot(fig)

    corr = df[an.numericas + [t]].corr()[t].drop(t)
    mayor = corr.abs().idxmax()

    st.subheader("✅ 5 conclusiones orientadas a decisiones")
    st.write(f"**1. El uso intensivo de redes es la señal más visible.** El grupo con etiqueta 1 "
             f"usa redes {redes1:.1f} h/día frente a {redes0:.1f} h. Entre quienes usan 6 h o "
             f"más, la proporción con etiqueta 1 es {tasa_alta:.1f}% frente a {tasa_baja:.1f}% "
             f"en quienes usan menos. *Evidencia: Ítem 7 y gráfico resumen.* "
             f"→ **Decisión:** priorizar programas de uso consciente en el tramo de mayor exposición.")
    st.write(f"**2. El sueño corto acompaña a la etiqueta.** Promedio de {sueno1:.1f} h frente a "
             f"{sueno0:.1f} h. *Evidencia: Ítem 7 (sleep_hours).* "
             f"→ **Decisión:** incluir higiene del sueño en las charlas, no solo el tiempo de pantalla.")
    st.write(f"**3. Estrés y ansiedad se separan tanto como redes y sueño.** Estrés {estres1:.1f} "
             f"vs {estres0:.1f} y ansiedad {ansiedad1:.1f} vs {ansiedad0:.1f}; en cambio "
             f"dependencia, actividad física y rendimiento académico casi no se distinguen. "
             f"*Evidencia: gráfico resumen.* → **Decisión:** usar estas escalas como señal de "
             f"alerta para derivar a orientación profesional, nunca como diagnóstico.")
    st.write("**4. Plataforma, género e interacción social casi no discriminan.** Las tasas por "
             "categoría son parecidas y con muy pocos casos por grupo. *Evidencia: Ítem 8.* "
             "→ **Decisión:** no segmentar campañas por plataforma o género con estos datos.")
    st.write(f"**5. Cuidado al interpretar: solo hay {n1} casos con etiqueta 1 "
             f"({porcentaje1:.1f}%).** La correlación más alta con la etiqueta es "
             f"{mayor} ({corr[mayor]:.2f}) y describe asociación, no causalidad. "
             f"*Evidencia: Ítems 6 y 9.* → **Decisión:** antes de intervenir, validar con una "
             f"muestra más grande y con profesionales de salud.")
    st.caption("Análisis exploratorio con fines educativos. No sustituye la valoración de profesionales de la salud.")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="EDA Teen Mental Health", page_icon="🧠", layout="wide")
    st.sidebar.title("🧭 Navegación")
    pagina = st.sidebar.radio("Módulo", ["🏠 Home", "📥 Carga del dataset", "📊 EDA"])
    if "df" in st.session_state:
        st.sidebar.success(f"Dataset cargado: {len(st.session_state['df'])} filas")
    else:
        st.sidebar.info("Sin dataset cargado")

    if pagina == "🏠 Home":
        pagina_home()
    elif pagina == "📥 Carga del dataset":
        pagina_carga()
    else:
        pagina_eda()


main()
