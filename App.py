##----- TRABAJO FINAL - PYTHON FOR ANALITYCS -----------
# Importamos las Librerias -------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generamos la CLASE: DATA ANALYZER ------
class DataAnalyzer:
    #Guardamos el DF como un atributo
    def __init__(self, df):
        self.df = df
    # Función para devolver variables numericas
    def numeric_cols(self):
        return self.df.select_dtypes(include=np.number).columns.tolist()
    # Función para devolver variables tipo texto  (Categóricas)
    def categorical_cols(self):
        return self.df.select_dtypes(exclude=np.number).columns.tolist()
    # Función para datos estadísticos del DF
    def stats(self):
        return self.df.describe()
    # Función para datos nulos o vacios
    def nulls(self):
        return self.df.isnull().sum()
    # Función para Media Agrupada del DF
    def group_mean(self, num, cat):
        return self.df.groupby(cat)[num].mean()

#------- SIDEBAR: MENÚ SELECCIONABLE  --------------
st.sidebar.title("Módulos")
modulo = st.sidebar.radio("Ir a:", ["Home", "Carga Dataset", "EDA", "Conclusiones"])

#------------- MÓDULO 1: HOME ------------- 
if modulo == "Home":
    #Título 
    st.title("Análisis Exploratorio - Bank Marketing")
    #Objetivo
    st.write('Análisis Exploratorio de Datos (EDA) aplicado al dataset BankMarketing.')
    #Subtitulo: Autor y Curso
    st.subheader("Autor")
    st.write("Diego Alfredo Chunga Bonilla")
    st.write("Especialización: Python for Analytics")
    st.write("2026")
    #Subtitulo: Descripción de Dataset   
    st.subheader("Descripción del dataset")
    st.write("Datos de campañas de marketing bancario.")
    #Subtitulo: Librerías 
    st.subheader("Tecnologías")
    st.write("Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit")

#------------- MÓDULO 2 : CARGA DATASET -----------

elif modulo == "Carga Dataset":
    #Nombre de la sección
    st.title("Carga del Dataset")
    # Carga de Archivo
    archivo = st.file_uploader("Sube el archivo en formato CSV", type=["csv"])
    # Formato de Archivo  (Separador ";")
    if archivo:
        df = pd.read_csv(archivo, sep=";")
        #Realizamos la limpieza de datos "Unknown o 999"
        df.replace(999, np.nan, inplace = True)
        df = df.applymap(lambda x: np.nan if str(x).lower()=="unknown" else x)
        st.success("Archivo cargado correctamente")

        #Forma del DF (Filas y Columnas)
        st.write("Dimensiones:", df.shape)
        st.dataframe(df.head())
        #Guardamos en la memora el df cargado
        st.session_state["df"] = df
    else:
        #Alerta cuando no se carga un archivo
        st.warning("No ha cargado un archivo")

#----- MODULO 3: ANÁLISIS EXPLORATORIO - EDA ---------
elif modulo == "EDA":
    #Advertencia en caso el DF no haya sido cargado
    if "df" not in st.session_state:
        st.warning("Cargue el dataset")
        st.stop()

    df = st.session_state["df"]
    analyzer = DataAnalyzer(df)

    st.title("Análisis Exploratorio")
    #Tablas de Análisis
    tabs = st.tabs([
        "Info General",
        "Clasificación",
        "Descriptivas",
        "Nulos",
        "Numéricas",
        "Categóricas",
        "Análisis - Numérica vs Categórica",
        "Análisis - Categórica vs Categórica",
        "Análisis de parámetros seleccionados",
        "Hallazgos"])

    # ITEM 1 : Información General del Dataset
    #Tipo de Datos y valores nulos
    with tabs[0]:
        st.subheader("Información General")
        st.write(df.dtypes)
        st.write("Valores nulos")
        st.write(analyzer.nulls())

    # ITEM 2 : Clasificación de variables
    # Identifica variables numericas y categóricas 
    with tabs[1]:
        st.subheader("Clasificación de Variables")
        st.write("Numéricas:", analyzer.numeric_cols())
        st.write("Categóricas:", analyzer.categorical_cols())

    # ITEM 3 : Estadística descriptiva
    with tabs[2]:
        st.subheader("Estadísticas Descriptivas")
        st.write(analyzer.stats())

    # ITEM 4 : Analisis de valores faltantes
    with tabs[3]:
        st.subheader("Valores Faltantes")
        st.write(analyzer.nulls())

    # ITEM 5: Distribución de variables numéricas
    with tabs[4]:
        st.subheader("Distribución Numéricas")
        #Seleccionamos variable
        col = st.selectbox("Seleccione variable", analyzer.numeric_cols())
        #Histograma de valores
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # ITEM 6: Análisis de variables categóricas
    with tabs[5]:
        st.subheader("Variables Categóricas")
        #Seleccionamos variable
        col = st.selectbox("Seleccione variable categórica", analyzer.categorical_cols())
        #Grafico de frecuencia / barras
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # ITEM 7: Análisis bivariado (numérico vs categórico)
    with tabs[6]:
        st.subheader("Numérico vs Categórico")

        num = st.selectbox("Variable numérica", analyzer.numeric_cols())
        fig, ax = plt.subplots()
        sns.boxplot(x="y", y=num, data=df, ax=ax)
        st.pyplot(fig)

    # ITEM 8: Análisis bivariado (categórico vs categórico)
    with tabs[7]:
        st.subheader("Categórico vs Categórico")

        cat = st.selectbox("Variable categórica", analyzer.categorical_cols())
        tabla = pd.crosstab(df[cat], df["y"])
        st.write(tabla)

    # ITEM 9: Análisis basado en parámetros seleccionados
    with tabs[8]:
        st.subheader("Análisis Dinámico")

        num = st.selectbox("Numérica", analyzer.numeric_cols(), key="n1")
        cat = st.selectbox("Categórica", analyzer.categorical_cols(), key="c1")

        st.write(analyzer.group_mean(num, cat))

    # ITEM 10: Hallazgos clave
    with tabs[9]:
        st.subheader("Hallazgos Clave")

        tasa = df["y"].value_counts(normalize=True)*100
        st.write("Tasa aceptación (%)")
        st.write(tasa)

        st.write("Insight: La aceptación es baja, se requiere segmentación.")



# CONCLUSIONES
elif modulo == "Conclusiones":

    st.title("Conclusiones Finales")

    st.write("""
    1. La tasa de aceptación es baja, por lo que ciertos factores deben ser analizados para priorizar recursos en aquellos que pueden aceptar la campaña.
    2. La duración del contacto influye significativamente.
    3. Algunos segmentos educativos muestran mayor respuesta, como por ejemplo "Hig-School" o "Universitarios".
    4. El canal de contacto impacta resultados. Si bien es cierto, por celular se acepta más campañas, hay un gran porcentaje que rechaza este medio de contacto
    5. Se recomienda optimizar segmentación comercial, haciendo enfasis en el día de la semana que se hace contacto.
    """)