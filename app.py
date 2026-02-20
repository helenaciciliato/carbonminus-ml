import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="CarbonMinus ML")

st.title("CarbonMinus - Machine Learning")
st.write("Modelo CatBoost treinado via PyCaret")

# Carregar modelo
model = load_model("carbonminus_catboost_intervalo_completo")

st.subheader("Inserir Condições Operacionais")

A = st.number_input("Temperatura (°C)", 40.0, 80.0)
B = st.number_input("Tempo (min)", 15.0, 45.0)
C = st.number_input("Razão H2O/CO2", 0.05, 2.00)

if st.button("Calcular captura estimada"):

    dados = pd.DataFrame({
        "A": [A],
        "B": [B],
        "C": [C]
    })

    pred = predict_model(model, data=dados)
    Y = pred["prediction_label"][0]

    if Y < 3:
        classificacao = "Baixa eficiência"
    elif Y < 5:
        classificacao = "Eficiência moderada"
    elif Y < 6:
        classificacao = "Alta eficiência"
    else:
        classificacao = "Muito alta eficiência"

    st.success(f"Captura estimada: {Y:.3f} mmol/g")
    st.write("Classificação:", classificacao)

    st.caption("Modelo: CatBoost Regressor | PyCaret")