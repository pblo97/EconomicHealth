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
    "BTFPAMOUNTS" : "Bank Term Funding Program Usage",

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

    "NONREVSL": "Nonrevolving Consumer Credit",
    "H8B1058NCBCAG": "Consumer Loans by Finance Companies",
    "H41RESPPALDKNWW": "Bank Term Funding Program Usage",
    "REVOLSL": "Revolving Consumer Credit",
    "DRCCLACBS": "Delinquency Rate on Credit Card Loans",
    "DRSFRMACBS": "Delinquency Rate on Single-Family Mortgages",

    
    "IPX5VHT2N": "Industrial Production: Total Excluding Selected High-Tech and Motor Vehicles & Parts",
    "PNFIC1": "Private Nonresidential Fixed Investment",  # Inversión privada no residencial
    "IPBUSEQ": "Industrial Production: Business Equipment",  # Producción de bienes de capital
    "IPG3254S": "Industrial Production: Pharmaceuticals and Medicine",  # Producción farmacéutica
    "IPUEN334413T011000000": "Real Sectoral Output for Manufacturing: Semiconductor and Related Device Manufacturing",  # Semiconductores
    "IPG3341S": "Industrial Production: Computers and Peripheral Equipment",  # Computadores
    "CES6562440001": "All Employees: Scientific Research and Development Services",  # Servicios de I+D
    "HOUST5F": "New Privately-Owned Housing Units Started: 5 Units or More" , # Innovación en viviendas multifamiliares

    "GDPC1": "Real Gross Domestic Product",
    "A191RL1Q225SBEA": "Real GDP per Capita",
    "GPDIC1": "Gross Private Domestic Investment",
    "TCU": "Capacity Utilization: Total Industry",
    "GDPPOT": "Real Potential Gross Domestic Product",
    "GCEC1": "Real Government Consumption Expenditures and Gross Investment",
    "NETEXP": "Net Exports of Goods and Services",
    "FYFSGDA188S": "Federal Debt to GDP Ratio",
    "PSAVERT": "Personal Saving Rate",

    "UMCSENT": "Consumer Sentiment (UMich)",
    "T5YIFR": "5Y5Y Forward Inflation Expectation",
    "T10YIE": "10Y Breakeven Inflation",
    "EXPINF1YR": "Expected Inflation 1Y Ahead",

    "GCEC1": "Real Government Consumption & Gross Investment",  # Gasto e inversión pública real
    "GFDEBTN": "Federal Debt: Total Public Debt",               # Deuda pública total nominal
    "GFDEGDQ188S": "Federal Debt: % of GDP",                    # Deuda federal como % del PIB
    "A091RC1Q027SBEA": "Interest Payments by the Federal Government",  # Pagos de intereses
    "MTSDS133FMS": "Federal Surplus or Deficit",

    "PCE": "Personal Consumption Expenditures",
    "PCECC96": "Real Personal Consumption Expenditures",
    "REVOLSL": "Consumer Credit: Credit Cards and Other Revolving Plans",
    "NONREVSL": "Consumer Credit: Nonrevolving",
          
    "PCEPI": "PCE: Chain-type Price Index",
    "WPUFD49207": "PPI: Intermediate Demand by Production Flow",
    "WPSFD49207": "PPI: Final Demand by Production Flow",
    "GFDEBTN": "Federal Debt: Total Public Debt",
         


    
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
    ["General", "🫀 Circulatorio", "🧠 Nervioso", "🫁 Pulmones", "🧬 Metabolismo", "🧪 Inmunológico (Shadow banking)", "Musculatura (Produccion industrial)", "Cadenas logisticas", "Higado (Sistema bancario)", "Utero (Innovacion y desarollo)", "Sistema Oseo(Estructura economica)","Comunicacion(Sentimiento de mercado)","Sistema autonomo(Politica fiscal y monetaria)","Sistema digestivo(consumo)","Temperatura(Inflacion)"]
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
    
    fig23 = go.Figure()
    fig23.add_trace(go.Scatter(x=df_full.index, y=df_full['Bank Term Funding Program Usage'], name="Bank Term Funding Program Usage", list = dict(color = "#36375D")))
    fig23.update_layout(title = "Bank Term Funding Program Usage", template = "plotly_dark", height = 800)
    st.plotly_chart(fig23, use_container_width=True)

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

elif sistema == "Higado (Sistema bancario)":
    st.header("Higado (sistema bancario)")

    fig_liv1 = go.Figure()
    fig_liv1.add_trace(go.Scatter(
    x=df_full.index,y=df_full["Nonrevolving Consumer Credit"],name="Crédito No Revolvente",line=dict(color="#8BC34A")))
    fig_liv1.update_layout(title="Nonrevolving Consumer Credit", template="plotly_dark", height=400)
    st.plotly_chart(fig_liv1, use_container_width=True)

# Consumer Loans by Finance Companies
    fig_liv2 = go.Figure()
    fig_liv2.add_trace(go.Scatter(x=df_full.index,y=df_full["Consumer Loans by Finance Companies"],name="Préstamos por Compañías Financieras",line=dict(color="#00ACC1")))
    fig_liv2.update_layout(title="Consumer Loans by Finance Companies", template="plotly_dark", height=400)
    st.plotly_chart(fig_liv2, use_container_width=True)

# Bank Term Funding Program (BTFP) Usage
    fig_liv3 = go.Figure()
    fig_liv3.add_trace(go.Scatter(x=df_full.index,y=df_full["Bank Term Funding Program Usage"],name="Uso del BTFP",line=dict(color="#FF8F00")))
    fig_liv3.update_layout(title="Bank Term Funding Program Usage (BTFP)", template="plotly_dark", height=400)
    st.plotly_chart(fig_liv3, use_container_width=True)

    fig_liv4 = go.Figure()
    fig_liv4.add_trace(go.Scatter(x=df_full.index,y=df_full["Revolving Consumer Credit"],name="Crédito Revolvente",line=dict(color="#AB47BC")))
    fig_liv4.update_layout(title="Revolving Consumer Credit", template="plotly_dark", height=400)
    st.plotly_chart(fig_liv4, use_container_width=True)

# Delinquency Rate - Credit Card Loans
    fig_liv5 = go.Figure()
    fig_liv5.add_trace(go.Scatter(x=df_full.index,y=df_full["Delinquency Rate on Credit Card Loans"],name="Morosidad Tarjetas de Crédito",line=dict(color="#EF5350")))
    fig_liv5.update_layout(title="Delinquency Rate - Credit Card Loans", template="plotly_dark", height=400)
    st.plotly_chart(fig_liv5, use_container_width=True)

# Delinquency Rate - Mortgages
    fig_liv6 = go.Figure()
    fig_liv6.add_trace(go.Scatter(x=df_full.index, y=df_full["Delinquency Rate on Single-Family Mortgages"],name="Morosidad Hipotecas Residenciales",line=dict(color="#5C6BC0")))
    fig_liv6.update_layout(title="Delinquency Rate - Single-Family Mortgages", template="plotly_dark", height=400)
    st.plotly_chart(fig_liv6, use_container_width=True)


elif sistema == "Utero (Innovacion y desarollo)":
    st.header("Utero (Innovaciony desarollo)")

    # Gráfico 1: PNFI - Private Nonresidential Fixed Investment
    fig_u1 = go.Figure()
    fig_u1.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production: Total Excluding Selected High-Tech and Motor Vehicles & Parts"],name="Excl. High-Tech & Autos", line=dict(color="#009688")))
    fig_u1.update_layout(title="Producción Industrial (excluyendo High-Tech y Autos)", template="plotly_dark", height=400)
    st.plotly_chart(fig_u1, use_container_width=True)

    fig_u2 = go.Figure()
    fig_u2.add_trace(go.Scatter(x=df_full.index, y=df_full["Private Nonresidential Fixed Investment"],name="Inversión Fija No Residencial", line=dict(color="#3F51B5")))
    fig_u2.update_layout(title="Inversión Fija No Residencial", template="plotly_dark", height=400)
    st.plotly_chart(fig_u2, use_container_width=True)

    fig_u3 = go.Figure()
    fig_u3.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production: Business Equipment"],name="Equipamiento Empresarial", line=dict(color="#FF5722")))
    fig_u3.update_layout(title="Producción de Bienes de Capital", template="plotly_dark", height=400)
    st.plotly_chart(fig_u3, use_container_width=True)

    fig_u4 = go.Figure()
    fig_u4.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production: Pharmaceuticals and Medicine"],name="Farmacéutica", line=dict(color="#8BC34A")))
    fig_u4.update_layout(title="Producción Farmacéutica", template="plotly_dark", height=400)
    st.plotly_chart(fig_u4, use_container_width=True)

    fig_u5 = go.Figure()
    fig_u5.add_trace(go.Scatter(x=df_full.index, y=df_full["Real Sectoral Output for Manufacturing: Semiconductor and Related Device Manufacturing"],name="Semiconductores", line=dict(color="#E91E63")))
    fig_u5.update_layout(title="Producción de Semiconductores", template="plotly_dark", height=400)
    st.plotly_chart(fig_u5, use_container_width=True)

    fig_u6 = go.Figure()
    fig_u6.add_trace(go.Scatter(x=df_full.index, y=df_full["Industrial Production: Computers and Peripheral Equipment"],name="Computadores", line=dict(color="#607D8B")))
    fig_u6.update_layout(title="Producción de Computadores", template="plotly_dark", height=400)
    st.plotly_chart(fig_u6, use_container_width=True)

    fig_u7 = go.Figure()
    fig_u7.add_trace(go.Scatter(x=df_full.index, y=df_full["All Employees: Scientific Research and Development Services"],name="Empleo en I+D", line=dict(color="#FFC107")))
    fig_u7.update_layout(title="Empleo en Servicios de I+D", template="plotly_dark", height=400)
    st.plotly_chart(fig_u7, use_container_width=True)

    fig_u8 = go.Figure()
    fig_u8.add_trace(go.Scatter(x=df_full.index, y=df_full["New Privately-Owned Housing Units Started: 5 Units or More"],name="Viviendas Multifamiliares", line=dict(color="#9C27B0")))
    fig_u8.update_layout(title="Innovación en Viviendas Multifamiliares", template="plotly_dark", height=400)
    st.plotly_chart(fig_u8, use_container_width=True)

elif sistema == "Sistema Oseo(Estructura economica)":
    st.header("Sistema Oseo (Estructura económica)")

    fig_s1 = go.Figure()
    fig_s1.add_trace(go.Scatter(x=df_full.index, y=df_full["Real Gross Domestic Product"], name="Real GDP", line=dict(color="#2196F3")))
    fig_s1.update_layout(title="Real Gross Domestic Product", template="plotly_dark", height=400)
    st.plotly_chart(fig_s1, use_container_width=True)

    fig_s2 = go.Figure()
    fig_s2.add_trace(go.Scatter(x=df_full.index, y=df_full["Real GDP per Capita"], name="GDP per Capita", line=dict(color="#4CAF50")))
    fig_s2.update_layout(title="Real GDP per Capita", template="plotly_dark", height=400)
    st.plotly_chart(fig_s2, use_container_width=True)

    fig_s3 = go.Figure()
    fig_s3.add_trace(go.Scatter(x=df_full.index, y=df_full["Gross Private Domestic Investment"], name="Private Investment", line=dict(color="#FFC107")))
    fig_s3.update_layout(title="Gross Private Domestic Investment", template="plotly_dark", height=400)
    st.plotly_chart(fig_s3, use_container_width=True)

    fig_s4 = go.Figure()
    fig_s4.add_trace(go.Scatter(x=df_full.index, y=df_full["Capacity Utilization: Total Industry"], name="Capacity Utilization", line=dict(color="#E91E63")))
    fig_s4.update_layout(title="Capacity Utilization", template="plotly_dark", height=400)
    st.plotly_chart(fig_s4, use_container_width=True)

    fig_s5 = go.Figure()
    fig_s5.add_trace(go.Scatter(x=df_full.index, y=df_full["Real Potential Gross Domestic Product"], name="Real Potential Gross Domestic Product", line=dict(color="#FF5722")))
    fig_s5.update_layout(title="Real Potential Gross Domestic Product", template="plotly_dark", height=400)
    st.plotly_chart(fig_s5, use_container_width=True)

    fig_s6 = go.Figure()
    fig_s6.add_trace(go.Scatter(x=df_full.index, y=df_full["Real Government Consumption Expenditures and Gross Investment"], name="Gov Investment", line=dict(color="#9C27B0")))
    fig_s6.update_layout(title="Gross Government Investment", template="plotly_dark", height=400)
    st.plotly_chart(fig_s6, use_container_width=True)

    fig_s7 = go.Figure()
    fig_s7.add_trace(go.Scatter(x=df_full.index, y=df_full["Net Exports of Goods and Services"], name="Net Exports", line=dict(color="#00BCD4")))
    fig_s7.update_layout(title="Net Exports of Goods and Services", template="plotly_dark", height=400)
    st.plotly_chart(fig_s7, use_container_width=True)

    fig_s8 = go.Figure()
    fig_s8.add_trace(go.Scatter(x=df_full.index, y=df_full["Federal Debt to GDP Ratio"], name="Debt to GDP", line=dict(color="#8BC34A")))
    fig_s8.update_layout(title="Federal Debt to GDP Ratio", template="plotly_dark", height=400)
    st.plotly_chart(fig_s8, use_container_width=True)

    fig_s9 = go.Figure()
    fig_s9.add_trace(go.Scatter(x=df_full.index, y=df_full["Personal Saving Rate"], name="Savings Rate", line=dict(color="#795548")))
    fig_s9.update_layout(title="Personal Saving Rate", template="plotly_dark", height=400)
    st.plotly_chart(fig_s9, use_container_width=True)

elif sistema == "Comunicacion(Sentimiento de mercado)":
    st.header("Comunicación (Sentimiento de mercado)")
    fig_c1 = go.Figure()
    fig_c1.add_trace(go.Scatter(x=df_full.index, y=df_full["Consumer Sentiment (UMich)"],name="Consumer Sentiment (UMich)", line=dict(color="#42A5F5")))
    fig_c1.update_layout(title="Consumer Sentiment (UMich)", template="plotly_dark", height=400)
    st.plotly_chart(fig_c1, use_container_width=True)

    # fig_c2: 5Y5Y Forward Inflation Expectation
    fig_c2 = go.Figure()
    fig_c2.add_trace(go.Scatter(x=df_full.index, y=df_full["5Y5Y Forward Inflation Expectation"],name="5Y5Y Forward Inflation Expectation", line=dict(color="#66BB6A")))
    fig_c2.update_layout(title="5Y5Y Forward Inflation Expectation", template="plotly_dark", height=400)
    st.plotly_chart(fig_c2, use_container_width=True)

    # fig_c3: 10Y Breakeven Inflation
    fig_c3 = go.Figure()
    fig_c3.add_trace(go.Scatter(x=df_full.index, y=df_full["10Y Breakeven Inflation"],name="10Y Breakeven Inflation", line=dict(color="#FFA726")))
    fig_c3.update_layout(title="10Y Breakeven Inflation", template="plotly_dark", height=400)
    st.plotly_chart(fig_c3, use_container_width=True)

    # fig_c4: Small Business Optimism Index (NFIB)
    fig_c4 = go.Figure()
    fig_c4.add_trace(go.Scatter(x=df_full.index, y=df_full["VIX"], name = "VIX", line = dict(color = "#431818")))
    fig_c4.update_layout(title="VIX", template="plotly_dark", height=400)
    st.plotly_chart(fig_c4, use_container_width=True)

    # fig_c5: Expected Inflation 1Y Ahead
    fig_c5 = go.Figure()
    fig_c5.add_trace(go.Scatter(x=df_full.index, y=df_full["Expected Inflation 1Y Ahead"],name="Expected Inflation 1Y Ahead", line=dict(color="#EF5350")))
    fig_c5.update_layout(title="Expected Inflation 1Y Ahead", template="plotly_dark", height=400)
    st.plotly_chart(fig_c5, use_container_width=True)

elif sistema == "Sistema autonomo(Politica fiscal y monetaria)":
    st.header("Sistema Autónomo (Política Fiscal y Monetaria)")

    fig_a1 = go.Figure()
    fig_a1.add_trace(go.Scatter(x=df_full.index, y=df_full["Real Government Consumption & Gross Investment"], name="Gasto e inversión pública", line=dict(color="#4CAF50")))
    fig_a1.update_layout(title="Gasto e Inversión Pública (Real)", template="plotly_dark", height=400)
    st.plotly_chart(fig_a1, use_container_width=True)

    fig_a2 = go.Figure()
    fig_a2.add_trace(go.Scatter(x=df_full.index, y=df_full["Federal Debt: Total Public Debt"], name="Deuda Federal Total", line=dict(color="#F44336")))
    fig_a2.update_layout(title="Deuda Pública Total", template="plotly_dark", height=400)
    st.plotly_chart(fig_a2, use_container_width=True)

    fig_a3 = go.Figure()
    fig_a3.add_trace(go.Scatter(x=df_full.index, y=df_full["Federal Debt: % of GDP"], name="Deuda como % del PIB", line=dict(color="#2196F3")))
    fig_a3.update_layout(title="Deuda Pública como % del PIB", template="plotly_dark", height=400)
    st.plotly_chart(fig_a3, use_container_width=True)

    fig_a4 = go.Figure()
    fig_a4.add_trace(go.Scatter(x=df_full.index, y=df_full["Interest Payments by the Federal Government"], name="Pagos de intereses", line=dict(color="#FFC107")))
    fig_a4.update_layout(title="Pagos de Interés del Gobierno Federal", template="plotly_dark", height=400)
    st.plotly_chart(fig_a4, use_container_width=True)

    fig_a5 = go.Figure()
    fig_a5.add_trace(go.Scatter(x=df_full.index, y=df_full["Federal Surplus or Deficit"], name="Superávit o Déficit Fiscal", line=dict(color="#9C27B0")))
    fig_a5.update_layout(title="Superávit o Déficit Fiscal", template="plotly_dark", height=400)
    st.plotly_chart(fig_a5, use_container_width=True)

elif sistema == "Sistema digestivo(consumo)":
    st.header("Sistema Digestivo (Consumo)")

    fig_d1 = go.Figure()
    fig_d1.add_trace(go.Scatter(x=df_full.index, y=df_full["Personal Consumption Expenditures"],name="PCE", line=dict(color="#29B6F6")))
    fig_d1.update_layout(title="Personal Consumption Expenditures", template="plotly_dark", height=400)
    st.plotly_chart(fig_d1, use_container_width=True)

    # Gasto real (PCECC96)
    fig_d2 = go.Figure()
    fig_d2.add_trace(go.Scatter(x=df_full.index, y=df_full["Real Personal Consumption Expenditures"], name="PCE Real", line=dict(color="#66BB6A")))
    fig_d2.update_layout(title="Real Personal Consumption Expenditures", template="plotly_dark", height=400)
    st.plotly_chart(fig_d2, use_container_width=True)

    # Crédito rotativo (tarjetas)
    fig_d3 = go.Figure()
    fig_d3.add_trace(go.Scatter(x=df_full.index, y=df_full["Consumer Credit: Credit Cards and Other Revolving Plans"],name="Revolving Credit", line=dict(color="#FF7043")))
    fig_d3.update_layout(title="Revolving Consumer Credit (Credit Cards)", template="plotly_dark", height=400)
    st.plotly_chart(fig_d3, use_container_width=True)

    # Crédito no rotativo
    fig_d4 = go.Figure()
    fig_d4.add_trace(go.Scatter(x=df_full.index, y=df_full["Consumer Credit: Nonrevolving"],name="Nonrevolving Credit", line=dict(color="#AB47BC")))
    fig_d4.update_layout(title="Nonrevolving Consumer Credit", template="plotly_dark", height=400)
    st.plotly_chart(fig_d4, use_container_width=True)

elif sistema == "Temperatura(Inflacion)":
    st.header("Temperatura (Inflación)")

    fig_temp1 = go.Figure()
    fig_temp1.add_trace(go.Scatter(x=df_full.index, y=df_full["PCE: Chain-type Price Index"], name="PCE Price Index", line=dict(color="#F44336")))
    fig_temp1.update_layout(title="PCE: Chain-type Price Index", template="plotly_dark", height=400)
    st.plotly_chart(fig_temp1, use_container_width=True)

    fig_temp2 = go.Figure()
    fig_temp2.add_trace(go.Scatter(x=df_full.index, y=df_full["PPI: Intermediate Demand by Production Flow"], name="PPI - Intermediate Demand", line=dict(color="#FF9800")))
    fig_temp2.update_layout(title="PPI - Intermediate Demand", template="plotly_dark", height=400)
    st.plotly_chart(fig_temp2, use_container_width=True)

    fig_temp3 = go.Figure()
    fig_temp3.add_trace(go.Scatter(x=df_full.index, y=df_full["PPI: Final Demand by Production Flow"], name="PPI - Final Demand", line=dict(color="#FFC107")))
    fig_temp3.update_layout(title="PPI - Final Demand", template="plotly_dark", height=400)
    st.plotly_chart(fig_temp3, use_container_width=True)

    fig_temp4 = go.Figure()
    fig_temp4.add_trace(go.Scatter(x=df_full.index, y=df_full["Federal Debt: Total Public Debt"], name="Total Public Debt", line=dict(color="#D32F2F")))
    fig_temp4.update_layout(title="Total Public Debt", template="plotly_dark", height=400)
    st.plotly_chart(fig_temp4, use_container_width=True)





    








    







    

    




    





