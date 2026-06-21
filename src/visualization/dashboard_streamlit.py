from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px


def cargar_predicciones():
    raiz = Path(__file__).resolve().parents[2]

    carpeta_predicciones = raiz / "results" / "predicciones_recesion"
    archivos_csv = list(carpeta_predicciones.glob("part-*.csv"))

    if archivos_csv:
        df = pd.read_csv(archivos_csv[0])
    else:
        ruta_powerbi = raiz / "dashboard_powerbi" / "predicciones_recesion_powerbi.csv"
        if ruta_powerbi.exists():
            df = pd.read_csv(ruta_powerbi, sep=";", decimal=",")
        else:
            st.error("No se encontró el archivo de predicciones.")
            st.stop()

    df["date"] = pd.to_datetime(df["date"])

    df["recession"] = df["recession"].astype(int)
    df["target_recession_3m"] = df["target_recession_3m"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["prob_recession"] = df["prob_recession"].astype(float)

    return df


st.set_page_config(
    page_title="Dashboard Recesiones Big Data",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-title {
        font-size: 16px;
        color: #555;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1f4e79;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df = cargar_predicciones()

st.title("Sistema de alerta temprana de recesiones económicas")
st.write(
    "Dashboard generado a partir de predicciones realizadas con SparkML "
    "utilizando indicadores macroeconómicos históricos de FRED-MD y la variable USREC."
)


# TARJETAS
total_registros = len(df)
recesiones_reales = int(df["target_recession_3m"].sum())
recesiones_predichas = int(df["prediction"].sum())
prob_promedio = df["prob_recession"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Total de registros</div>
            <div class="metric-value">{total_registros}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Recesiones reales</div>
            <div class="metric-value">{recesiones_reales}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Recesiones predichas</div>
            <div class="metric-value">{recesiones_predichas}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Probabilidad promedio</div>
            <div class="metric-value">{prob_promedio:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# GRÁFICAS PRINCIPALES
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Probabilidad estimada de recesión económica")

    fig_prob = px.line(
        df,
        x="date",
        y="prob_recession",
        labels={
            "date": "Fecha",
            "prob_recession": "Probabilidad de recesión"
        },
        title="Probabilidad de recesión estimada por el modelo"
    )

    st.plotly_chart(fig_prob, use_container_width=True)

with col_g2:
    st.subheader("Recesión real vs predicción del modelo")

    df_comp = df.copy()
    df_comp["Recesión real"] = df_comp["target_recession_3m"]
    df_comp["Predicción del modelo"] = df_comp["prediction"]

    fig_comp = px.line(
        df_comp,
        x="date",
        y=["Recesión real", "Predicción del modelo"],
        labels={
            "date": "Fecha",
            "value": "Valor",
            "variable": "Serie"
        },
        title="Comparación entre valor real y predicción"
    )

    st.plotly_chart(fig_comp, use_container_width=True)

st.divider()


# MATRIZ DE CONFUSIÓN
st.subheader("Matriz de confusión del modelo")

matriz = pd.crosstab(
    df["target_recession_3m"],
    df["prediction"],
    rownames=["Valor real"],
    colnames=["Predicción"],
    dropna=False
)

col_m1, col_m2 = st.columns([1, 2])

with col_m1:
    st.dataframe(matriz, use_container_width=True)

with col_m2:
    fig_matriz = px.imshow(
        matriz,
        text_auto=True,
        title="Matriz de confusión",
        labels=dict(
            x="Predicción",
            y="Valor real",
            color="Cantidad"
        )
    )

    st.plotly_chart(fig_matriz, use_container_width=True)

st.divider()


# TABLA FINAL
st.subheader("Tabla final de predicciones")

df_tabla = df[
    [
        "date",
        "recession",
        "target_recession_3m",
        "prediction",
        "prob_recession"
    ]
].copy()

df_tabla["prob_recession"] = df_tabla["prob_recession"].map(lambda x: f"{x:.2%}")

st.dataframe(df_tabla, use_container_width=True)

st.divider()

st.subheader("Interpretación general")

st.write(
    "El dashboard muestra la probabilidad estimada de recesión económica en un horizonte "
    "de tres meses. La matriz de confusión permite observar el desempeño del modelo: "
    "aunque identifica correctamente la mayoría de los periodos sin recesión, los casos "
    "reales de recesión son pocos, lo que evidencia un desbalance de clases en los datos."
)