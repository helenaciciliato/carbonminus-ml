import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# CARREGAR MODELO
# -----------------------------
model = joblib.load("modelo_carbonminus.pkl")

st.title("CarbonMinus - Previsão de Captura de CO₂")
st.write("Modelo CatBoost treinado sem PyCaret (versão produção)")

st.subheader("Inserir Condições Operacionais")


A = st.number_input("Temperatura (°C)", min_value=40.0, max_value=80.0, value=61.0)
B = st.number_input("Tempo (min)", min_value=15.0, max_value=45.0, value=40.0)
C = st.number_input("Razão H2O/CO2", min_value=0.05, max_value=2.00, value=1.15)

if st.button("Calcular captura estimada"):

    dados = pd.DataFrame({
        "A": [A],
        "B": [B],
        "C": [C]
    })

    pred = model.predict(dados)
    Y = pred[0]

    st.success(f"Captura estimada: {Y:.3f} mmol/g")

    if Y < 3:
        st.info("Baixa eficiência")
    elif Y < 5:
        st.info("Eficiência moderada")
    elif Y < 6:
        st.info("Alta eficiência")
    else:
        st.info("Muito alta eficiência")