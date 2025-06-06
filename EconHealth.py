import os
import time
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from fredapi import Fred
import streamlit as st

st.set_page_config(layout="wide")

fred = Fred(api_key="02ea49012ba021ea89f1110c48de7380")

start = '2021-01-01'
end = dt.datetime.today()


series = {
    # 🟦 Tasas clave
    "SOFR": "SOFR",
    "EFFR": "EFFR",  # Effective Federal Funds Rate (ID oficial simplificado)
    "IORB": "Interest Rate on Reserve Balances",

    # 🟩 Curva de rendimiento y spreads
    "DGS2": "2Y Treasury Yield",
    "DGS10": "10Y Treasury Yield",
    "DTB3": "3M T-Bill Yield",
    "BAA10Y": "BAA Spread over 10Y Treasury",
    "BAMLC0A4CBBBEY": "BBB Corporate Bond Yield",

    # 🟧 Liquidez bancaria y de mercado
    "RRPONTSYD": "Reverse Repo (ON RRP)",
    "WALCL": "Fed Balance Sheet Total Assets",
    "WLRRAL": "Reserve Balances with Federal Reserve Banks",
    "M2SL": "M2 Money Stock",
    "M2V": "Velocity of M2 Money Stock",
    "TOTBKCR": "Total Bank Credit",
    "COMPOUT": "Commercial Paper Outstanding",

    # 🟥 Riesgo crediticio
    "BAMLH0A0HYM2": "High Yield Spread",

    # 🟨 Condiciones financieras
    "NFCI": "Chicago Fed Financial Conditions Index",

    # 🟪 Expectativas macroeconómicas
    "T5YIE": "5Y Breakeven Inflation Rate",
    "USRECD": "Recession Indicator",
    "ICSA": "Initial Jobless Claims",
    "TEDRATE": "TED Spread",

    # 🟫 Hogares y empleo
    "TNWBSHNO": "Household and Nonprofit Net Worth",
    "AWHAETP": "Avg Weekly Hours (Private Sector)",

    # 🟦 Adicionales útiles
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI - All Urban Consumers",
    "PCE": "Personal Consumption Expenditures",
    "HOUST": "Housing Starts",

    # 🧠 Volatilidad implícita
    "VIXCLS": "VIX"
}

data = {}

for series_id, name in series.items():
    try:
        data[name] = fred.get_series(series_id, start, end)
        print(f"Descargado: {series_id} - {name}")
    except ValueError as e:
        print(f"Error con la serie {series_id} - {name}: {e}")
    

df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)

df_full = df.resample('D').last().ffill().dropna()


df_full['SOFR-EFFR'] = (
    df_full["SOFR"] - 
    df_full["EFFR"]
)

df_corazon = df_full[[
    'SOFR-EFFR',
    'M2 Money Stock',
    'Reserve Balances with Federal Reserve Banks',
    'Reverse Repo (ON RRP)'
    

]]

df_full['2Y Treasury Yield - 10Y Treasury Yield'] = (
    df_full['2Y Treasury Yield'] - df_full['10Y Treasury Yield']
)

# Configuracion de la app

st.title("🧍 Anatomía Económica del Mercado")

sistema = st.sidebar.selectbox(
    "Selecciona un sistema",
    ["General", "🫀 Circulatorio", "🧠 Nervioso", "🫁 Pulmones", "🧬 Metabolismo"]
)

# ----------- SISTEMA CIRCULATORIO -------------- #
if sistema == "General":

    "Sistema de analisis de inversion"


elif sistema == "🫀 Circulatorio":
    
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_full.index, y=df_full['SOFR-EFFR'], name='SOFR - EFFR', line=dict(color="#7593A9")))
    fig1.update_layout(title="Spread SOFR vs EFFR", template="plotly_dark", height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_full.index, y=df_full["M2 Money Stock"], name='M2', line=dict(color="cyan")))
    fig2.update_layout(title="M2 Money Stock", template="plotly_dark", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_full.index, y=df_full["Reverse Repo (ON RRP)"], name='RRP', line=dict(color="orange")))
    fig3.update_layout(title="Reverse Repo", template="plotly_dark", height=400)
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_full.index, y=df_full["Reserve Balances with Federal Reserve Banks"], name='Reservas', line=dict(color="lightgreen")))
    fig4.update_layout(title="Reservas Bancarias", template="plotly_dark", height=400)
    st.plotly_chart(fig4, use_container_width=True) 


# ----------- SISTEMA NERVIOSO -------------- #
elif sistema == "🧠 Nervioso":
    st.header("🧠 Sistema Nervioso (Condiciones Financieras)")

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=df_full.index, y=df_full["Chicago Fed Financial Conditions Index"], name='NFCI', line=dict(color="orange")))
    fig5.update_layout(title="NFCI - Condiciones Financieras", template="plotly_dark", height=400)
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=df_full.index, y=df_full["VIX"], name='VIX', line=dict(color="violet")))
    fig6.update_layout(title="VIX - Volatilidad Esperada", template="plotly_dark", height=400)
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=df_full.index, y=df_full["High Yield Spread"], name='HY Spread', line=dict(color="salmon")))
    fig7.update_layout(title="High Yield Spread", template="plotly_dark", height=400)
    st.plotly_chart(fig7, use_container_width=True)

elif sistema == "🫁 Pulmones":
    st.header("Pulmones (Curva de rendimientos)")

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=df_full.index, y=df_full['2Y Treasury Yield - 10Y Treasury Yield'], name = '2Y - 10Y', line = dict(color = "red")))
    fig8.update_layout(title="Curva de Rendimientos", template="plotly_dark", height = 400)
    st.plotly_chart(fig8, use_container_width=True)

    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=df_full.index, y=df_full["BAA Spread over 10Y Treasury"], name = "BAA spread", line = dict(color = "violet")))
    fig9.update_layout(title="BAA Spread", template="plotly_dark", height =400)
    st.plotly_chart(fig9, use_container_width=True)

    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(x = df_full.index, y = df_full["Total Bank Credit"], name = "Credito bancario total", line = dict(color = "green") ))
    fig10.update_layout(title = "Credito", template = "plotly_dark", height = 800)
    st.plotly_chart(fig10, use_container_width=True)

elif sistema == "🧬 Metabolismo":
    st.header("Metabolismo (Empleo y actividad)")
    fig11 = go.Figure()
    fig11.add_trace(go.Scatter(x=df_full.index, y=df_full["Initial Jobless Claims"], name = "Desempleo", line = dict(color = "#845A5A")))
    fig11.update_layout(title = "Desempleo", template = "plotly_dark", height = 800)
    st.plotly_chart(fig11, use_container_width=True)

    fig12 = go.Figure()
    fig12.add_trace(go.Scatter(x=df_full.index, y=df_full["Personal Consumption Expenditures"], name = "Consumo", line = dict(color = "#555C78")))
    fig12.update_layout(title = "Consumo", template = "plotly_dark", height = 800)
    st.plotly_chart(fig12, use_container_width=True)

    fig13 = go.Figure()
    fig13.add_trace(go.Scatter(x=df_full.index, y=df_full["Avg Weekly Hours (Private Sector)"], name= "Promedio horas trabajadas sector privada", line = dict(color = "#B6A268")))
    fig13.update_layout(title = "Promedio horas trabajadas sector privada", template = "plotly_dark", height = 800)
    st.plotly_chart(fig13, use_container_width=True)

    





