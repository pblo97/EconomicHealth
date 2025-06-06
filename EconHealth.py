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
    "DAAA": "AAA Corporate Bond Yield",

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

    # Volatilidad implícita
    "VIXCLS": "VIX",

    
    "COMPOUT": "Commercial Paper Outstanding",
    "DRTSCILM": "Loan Officer Survey: % Banks Tightening C&I Loans (Large Firms)",
    "RIFSPPFAAD90NB": "Net Assets of Money Market Funds (AUM)",
    "H8B1058NCBCAG": "Consumer Loans by Finance Companies",
    "NONREVSL": "Nonrevolving Consumer Credit",

    #Produccion industrial

    "INDPRO": "Industrial Production Index",
    "TCU": "Capacity Utilization: Total Industry",
    "AMTMNO": "Manufacturers' New Orders: Total Manufacturing",
    "IPMAN": "Industrial Production: Manufacturing",
    "IPB50001N": "Industrial Production: Durable Consumer Goods",
    "IPG331S": "Industrial Production: Mining",
    "CMRMTSPL": "Real Manufacturing and Trade Sales",
    "BUSINV" : "Total Business Inventories",
    "MANEMP": "All Employees, Manufacturing",

    # 🦠 Sistema linfático: Cadenas logísticas
    
    "TSIFRGHT": "Freight Transportation Services Index",
    "WPU301": "Truck Transportation PPI",
    "ISRATIO": "Total Business: Inventories to Sales Ratio",
    

    
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

df_full['BBB-AAA Spread'] = (
    df_full['BBB Corporate Bond Yield'] - df_full['AAA Corporate Bond Yield']
)
# Configuracion de la app

st.title("🧍 Anatomía Económica del Mercado")

sistema = st.sidebar.selectbox(
    "Selecciona un sistema",
    ["General", "🫀 Circulatorio", "🧠 Nervioso", "🫁 Pulmones", "🧬 Metabolismo", "🧪 Inmunológico (Shadow banking)", "Musculatura (Produccion industrial)", "Cadenas logisticas"]
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
    fig11.add_trace(go.Scatter(x=df_full.index, y=df_full["Initial Jobless Claims"], name = "Initial Jobless Claims", line = dict(color = "#845A5A")))
    fig11.update_layout(title = "Solicitud seguro desempleo por primera vez en una seman", template = "plotly_dark", height = 800)
    st.plotly_chart(fig11, use_container_width=True)

    fig12 = go.Figure()
    fig12.add_trace(go.Scatter(x=df_full.index, y=df_full["Personal Consumption Expenditures"], name = "Personal Consumption Expenditure", line = dict(color = "#555C78")))
    fig12.update_layout(title = "Consumo", template = "plotly_dark", height = 800)
    st.plotly_chart(fig12, use_container_width=True)

    fig13 = go.Figure()
    fig13.add_trace(go.Scatter(x=df_full.index, y=df_full["Avg Weekly Hours (Private Sector)"], name= "Avg Weekly Hours (Private Sector)", line = dict(color = "#B6A268")))
    fig13.update_layout(title = "Promedio horas trabajadas sector privada", template = "plotly_dark", height = 800)
    st.plotly_chart(fig13, use_container_width=True)

    fig13_1 = go.Figure()
    fig13_1.add_trace(go.Scatter(x=df_full.index, y=df_full["All Employees, Manufacturing"], name = "Manufacturing Employment", line = dict(color = "#67642D")))
    fig13_1.update_layout(title = "Manufacturing Employment", template = "plotly_dark", height = 800)
    st.plotly_chart(fig13_1, use_container_width=True)

elif sistema == "🧪 Inmunológico (Shadow banking)":
    st.header("Inmunológico (Shadow banking)")
    fig14 = go.Figure()
    fig14.add_trace(go.Scatter(x=df_full.index, y=df_full["Commercial Paper Outstanding"], name = "Credito fuera del sistema", line = dict(color = "#60695F")))
    fig14.update_layout(title = "Credito fuera del sistema", template = "plotly_dark", height = 800)
    st.plotly_chart(fig14, use_container_width=True)

    fig15 = go.Figure()
    fig15.add_trace(go.Scatter(x=df_full.index, y=df_full['BBB-AAA Spread'], name = "'BBB-AAA Spread'", line = dict(color = "#3A1010")))
    fig15.update_layout(title = "'BBB-AAA Spread'", template = "plotly_dark", height = 800)
    st.plotly_chart(fig15, use_container_width=True)

    fig16 = go.Figure()
    fig16.add_trace(go.Scatter(x=df_full.index, y=df_full['Loan Officer Survey: % Banks Tightening C&I Loans (Large Firms)'], name="Loan Officer Survey: % Banks Tightening C&I Loans", line = dict(color = "#546086")))
    fig16.update_layout(title = "Loan Officer Survey: % Banks Tightening C&I Loans", template = "plotly_dark", height = 800)
    st.plotly_chart(fig16, use_container_width=True)

    fig17 = go.Figure()
    fig17.add_trace(go.Scatter(x=df_full.index, y=df_full['Fed Balance Sheet Total Assets'], name="Fed Balance Sheet Total Assets", line = dict(color = "#714955")))
    fig17.update_layout(title = "Fed Balance Sheet Total Assets", template = "plotly_dark", height = 800)
    st.plotly_chart(fig17, use_container_width=True)

    fig18 = go.Figure()
    fig18.add_trace(go.Scatter(x=df_full.index, y=df_full['Net Assets of Money Market Funds (AUM)'], name = "Net Assets of Money Market Funds (AUM)", line = dict(color = "#8D8758")))
    fig18.update_layout(title = "Net Assets of Money Market Funds (AUM)", template = "plotly_dark", height = 800)
    st.plotly_chart(fig18, use_container_width=True)

    fig20 = go.Figure()
    fig20.add_trace(go.Scatter(x=df_full.index, y=df_full["Reverse Repo (ON RRP)"], name='RRP', line=dict(color="orange")))
    fig20.update_layout(title="Reverse Repo", template="plotly_dark", height=400)
    st.plotly_chart(fig20, use_container_width=True)

    fig21 = go.Figure()
    fig21.add_trace(go.Scatter(x=df_full.index, y=df_full["Consumer Loans by Finance Companies"],name="Consumer Loans by Finance Companies", line = dict(color = "#373C5F")))
    fig21.update_layout(title="Consumer Loans by Finance Companies", template="plotly_dark", height = 800)
    st.plotly_chart(fig21, use_container_width=True)

    fig22 = go.Figure()
    fig22.add_trace(go.Scatter(x=df_full.index, y = df_full['Nonrevolving Consumer Credit'], name = "Nonrevolving Consumer Credit", line = dict(color = "#5D6531")))
    fig22.update_layout(title = "Nonrevolving Consumer Credit", template = "plotly_dark", height = 800)
    st.plotly_chart(fig22, use_container_width=True)

elif sistema == "Musculatura (Produccion industrial)":
    st.header("Musculatura (Produccion industrial)")
    # Gráfico 1: Índice de Producción Industrial (INDPRO)
    fig_m1 = go.Figure()
    fig_m1.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production Index"],name="Industrial Production Index", line=dict(color="#4CAF50")))
    fig_m1.update_layout(title="Producción Industrial Total", template="plotly_dark", height=400)
    st.plotly_chart(fig_m1, use_container_width=True)

    # Gráfico 2: Utilización de la Capacidad Instalada (TCU)
    fig_m2 = go.Figure()
    fig_m2.add_trace(go.Scatter(x=df_full.index, y=df_full["Capacity Utilization: Total Industry"],name="Capacidad Instalada", line=dict(color="#FFC107")))
    fig_m2.update_layout(title="Utilización de Capacidad Total", template="plotly_dark", height=400)
    st.plotly_chart(fig_m2, use_container_width=True)

    # Gráfico 3: PMI Manufacturero (NAPMPI)
    fig_m3 = go.Figure()
    fig_m3.add_trace(go.Scatter(x=df_full.index, y=df_full["Manufacturers' New Orders: Total Manufacturing"],name="Manufacturers' New Orders: Total Manufacturing", line=dict(color="#2196F3")))
    fig_m3.update_layout(title="Manufacturers' New Orders: Total Manufacturing", template="plotly_dark", height=400)
    st.plotly_chart(fig_m3, use_container_width=True)

    # Gráfico 4: Producción Manufacturera, Bienes duraderos y Minería
    fig_m4 = go.Figure()
    fig_m4.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production: Manufacturing"], name="Manufactura", line=dict(color="#E91E63")))
    fig_m4.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production: Durable Consumer Goods"],name="Bienes Duraderos", line=dict(color="#9C27B0")))
    fig_m4.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production: Mining"],name="Minería", line=dict(color="#00BCD4")))
    fig_m4.update_layout(title="Subcomponentes de Producción Industrial", template="plotly_dark", height=500)
    st.plotly_chart(fig_m4, use_container_width=True)

    fig_m5 = go.Figure()
    fig_m5.add_trace(go.Scatter(x=df_full.index, y=df_full["Real Manufacturing and Trade Sales"], name = "Real Manufacturing and Trade Sales", line = dict(color= "#72614F")))
    fig_m5.update_layout(title = "Real Manufacturing and Trade Sales", template = "plotly_dark", height = 800)
    st.plotly_chart(fig_m5, use_container_width=True)

    fig_m6 = go.Figure()
    fig_m6.add_trace(go.Scatter(x=df_full.index, y=df_full["Total Business Inventories"], name = "Total Business Inventories", line = dict(color = "#4A2D2D")))
    fig_m6.update_layout(title = "Total Business Inventories", template = "plotly_dark",height = 800)
    st.plotly_chart(fig_m6, use_container_width=True)

elif sistema == "Cadenas logisticas":
    st.header("Cadenas Logisticas")
    

    fig_l2 = go.Figure()
    fig_l2.add_trace(go.Scatter(x=df_full.index,y=df_full["Freight Transportation Services Index"],name="Freight TSI",line=dict(color="#F44336")))
    fig_l2.update_layout(title="Freight Transportation Services Index", template="plotly_dark", height=400)
    st.plotly_chart(fig_l2, use_container_width=True)

    fig_l3 = go.Figure()
    fig_l3.add_trace(go.Scatter(x=df_full.index,y=df_full["Truck Transportation PPI"],name="Truck PPI",line=dict(color="#9C27B0")))
    fig_l3.update_layout(title="Truck Transportation PPI", template="plotly_dark", height=400)
    st.plotly_chart(fig_l3, use_container_width=True)

    fig_l4 = go.Figure()
    fig_l4.add_trace(go.Scatter(x=df_full.index,y=df_full["Total Business: Inventories to Sales Ratio"],name="Inventories/Sales Ratio",line=dict(color="#4CAF50")))
    fig_l4.update_layout(title="Inventories to Sales Ratio", template="plotly_dark", height=400)
    st.plotly_chart(fig_l4, use_container_width=True)








    







    

    




    





