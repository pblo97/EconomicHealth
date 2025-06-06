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
    "SOFR": "SOFR (Secured Overnight Financing Rate)",
    "FEDFUNDS": "Effective Federal Funds Rate",
    "IORB": "Interest Rate on Reserve Balances",
    "EFFR": "Effective Federal Funds Rate (Alt)",

    # 🟩 Curva de rendimiento y spreads
    "DGS2": "2Y Treasury Yield",
    "DGS10": "10Y Treasury Yield",
    "DTB3": "3M T-Bill Yield",
    "BAA10Y": "BAA Spread over 10Y Treasury",
    "BAMLC0A4CBBBEY": "Yield on BBB Corporate Bonds",

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
    "USRECD": "Recession Indicator (NBER Binary)",
    "ICSA": "Initial Jobless Claims",
    "TEDRATE": "TED Spread (LIBOR - T-Bill 3M)", # 
     # 🟫 Hogares y empleo
    "TNWBSHNO": "Household and Nonprofit Net Worth",
    "AWHAETP": "Avg Weekly Hours of All Employees (Total Private)",
    
    # 🟦 Balance del Sistema de la Fed
    

    # (opcionalmente más)
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI - All Urban Consumers",
    "PCE": "Personal Consumption Expenditures",
    "HOUST": "Housing Starts",

    'VIXCLS': "VIX"
    
}

data = {}

for series_id, name in series.items():
    data[name] = fred.get_series(series_id, start, end)

df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)

df_full = df.resample('D').last().ffill().dropna()


df_full['SOFR-EFFR'] = (
    df_full["SOFR (Secured Overnight Financing Rate)"] - 
    df_full["Effective Federal Funds Rate (Alt)"]
)

# Configuracion de la app

st.title("🧍 Anatomía Económica del Mercado")

sistema = st.sidebar.selectbox(
    "Selecciona un sistema",
    ["🫀 Circulatorio", "🧠 Nervioso"]
)

# ----------- SISTEMA CIRCULATORIO -------------- #
if sistema == "🢀 Circulatorio":
    st.header("🫀 Sistema Circulatorio (Liquidez)")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_full.index, y=df_full['SOFR - EFFR'], name='SOFR - EFFR', line=dict(color="white")))
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
