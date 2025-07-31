import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Configuración inicial ---
st.set_page_config(
    page_title="Dashboard Energía España",
    page_icon="⚡",
    layout="wide"
)
st.title("🪫 Análisis y Predicción de Generación Eléctrica en España")
st.markdown("Modelado predictivo con Facebook Prophet y técnicas de machine learning para pronosticar la demanda energética y la generación renovable.")

# --- Carga datos y modelos ---
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv('../data/clean/energy_spain_clean.csv', parse_dates=['date'])
    return df

@st.cache_resource(show_spinner=True)
def load_models():
    model_prophet = joblib.load('../src/prophet_energy_model.joblib')
    model_rf = joblib.load('../src/rf_energy_model.joblib')
    return model_prophet, model_rf

df = load_data()
model_prophet, model_rf = load_models()

# --- Sidebar opciones ---
st.sidebar.header("Opciones de visualización y modelos")

model_option = st.sidebar.selectbox("Selecciona modelo predictivo", ["Prophet", "Random Forest"])

show_corr = st.sidebar.checkbox("Mostrar mapa de correlación", value=True)
show_imp = st.sidebar.checkbox("Mostrar importancia variables RF", value=True)
show_pred = st.sidebar.checkbox("Mostrar predicciones y comparativas", value=True)

# --- Funciones auxiliares ---

def plot_timeseries(df, cols, title):
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df['date'], y=df[c], mode='lines+markers', name=c))
    fig.update_layout(title=title, xaxis_title="Año", yaxis_title="GWh / Unidades")
    return fig

def calculate_rf_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# --- Visualización datos históricos ---
st.subheader("📈 Evolución histórica generación eléctrica")

cols_to_plot = ['electricity_generation', 'renewables_electricity', 'fossil_electricity']
fig1 = plot_timeseries(df, cols_to_plot, "Generación Eléctrica Total y por Tipo")
st.plotly_chart(fig1, use_container_width=True)

# --- Mapa de correlación ---
if show_corr:
    st.subheader("🔍 Mapa de correlación variables energéticas")
    corr = df.drop(columns=['year', 'date']).corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlación variables")
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Predicciones y comparación modelos ---
if show_pred:
    st.subheader("🤖 Predicciones: Modelo seleccionado")

    if model_option == "Prophet":
        st.markdown("Modelo **Prophet** para predicción de generación total eléctrica con intervalo de confianza.")

        # Preparar dataframe para Prophet
        df_prophet = df.rename(columns={'date':'ds', 'electricity_generation':'y'})[['ds','y']]
        future = model_prophet.make_future_dataframe(periods=10, freq='Y')
        forecast = model_prophet.predict(future)

        # Mostrar gráfico interactivo con intervalos
        fig_forecast = plot_plotly(model_prophet, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Métricas en datos entrenados (últimos 10 años)
        train_df = df_prophet.tail(10)
        pred_train = forecast.set_index('ds').loc[train_df['ds'], 'yhat']
        mae, mse, rmse = calculate_rf_metrics(train_df['y'], pred_train)
        st.markdown(f"**MAE** (últimos 10 años): {mae:.2f}")
        st.markdown(f"**RMSE** (últimos 10 años): {rmse:.2f}")

    else:  # Random Forest
        st.markdown("Modelo **Random Forest Regressor** para predicción multivariante de generación total eléctrica.")

        features = ['renewables_electricity', 'fossil_electricity', 'coal_electricity', 'gas_electricity', 
                    'nuclear_electricity', 'oil_electricity', 'hydro_electricity', 'solar_electricity', 
                    'wind_electricity', 'gdp', 'population']
        df_rf = df.dropna(subset=features + ['electricity_generation']).copy()
        X = df_rf[features]
        y_true = df_rf['electricity_generation']
        y_pred = model_rf.predict(X)

        comp_df = pd.DataFrame({'Real': y_true, 'Predicción': y_pred}, index=df_rf['year'])
        st.line_chart(comp_df)

        mae, mse, rmse = calculate_rf_metrics(y_true, y_pred)
        st.markdown(f"**MAE**: {mae:.2f}")
        st.markdown(f"**RMSE**: {rmse:.2f}")

        if show_imp:
            st.subheader("📊 Importancia de variables (Random Forest)")
            importances = model_rf.feature_importances_
            imp_df = pd.DataFrame({'Variable': features, 'Importancia': importances})
            imp_df = imp_df.sort_values(by='Importancia', ascending=False)
            fig_imp = px.bar(imp_df, x='Variable', y='Importancia', title='Importancia de Variables RF', text='Importancia')
            fig_imp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_imp, use_container_width=True)

# --- Análisis exploratorio adicional ---
st.markdown("---")
st.subheader("🔎 Análisis Exploratorio Detallado")

# Selección de variables para graficar
all_vars = list(df.columns)
var1 = st.selectbox("Selecciona variable eje X", all_vars, index=all_vars.index('year'))
var2 = st.selectbox("Selecciona variable eje Y", all_vars, index=all_vars.index('electricity_generation'))

fig_scatter = px.scatter(df, x=var1, y=var2, trendline="ols", title=f"Relación entre {var1} y {var2}")
st.plotly_chart(fig_scatter, use_container_width=True)

# Mostrar tabla resumen estadístico
st.markdown("### Estadísticas descriptivas de variables seleccionadas")
st.write(df[[var1,var2]].describe())

# --- Pie de página ---
st.markdown("---")
st.markdown("📌 Proyecto desarrollado por **Francisco Javier Gómez Pulido**")
st.markdown("📊 Datos: Our World in Data")
st.markdown("📜 Perfil de Linkedin: www.linkedin.com/in/frangomezpulido")
st.markdown("🔗 Repositorio de GitHub: https://github.com/fragompul/hospital_capacity_prediction.git")