import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from fredapi import Fred
import streamlit as st


class Sistema:
    """Base class that connects to the FRED API and builds a DataFrame."""

    API_KEY = "02ea49012ba021ea89f1110c48de7380"

    def __init__(self, start="2021-01-01", end=None):
        self.fred = Fred(api_key=self.API_KEY)
        self.start = start
        self.end = end or dt.datetime.today()
        self.series = {}
        self.data = None

    def fetch_data(self):
        data = {}
        for series_id, name in self.series.items():
            try:
                data[name] = self.fred.get_series(series_id, self.start, self.end)
            except Exception as exc:
                print(f"Error fetching {series_id}: {exc}")
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        self.data = df.resample("D").last().ffill().dropna()
        return self.data

    def prepare(self):
        """Hook for subclasses to add derived columns."""
        pass

    def get_data(self):
        if self.data is None:
            self.fetch_data()
            self.prepare()
        return self.data

    def create_plot(self, column, title, color):
        df = self.get_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column, line=dict(color=color)))
        fig.update_layout(title=title, template="plotly_dark", height=400)
        return fig

    def plots(self):
        return []


class Circulatorio(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "SOFR": "SOFR",
            "EFFR": "EFFR",
            "M2SL": "M2 Money Stock",
            "RRPONTSYD": "Reverse Repo (ON RRP)",
            "WLRRAL": "Reserve Balances with Federal Reserve Banks",
            "M2V": "Velocity of M2 Money Stock",
        }

    def prepare(self):
        df = self.data
        df['SOFR-EFFR'] = df['SOFR'] - df['EFFR']

    def plots(self):
        return [
            self.create_plot('SOFR-EFFR', 'Spread SOFR vs EFFR', '#7593A9'),
            self.create_plot('M2 Money Stock', 'M2 Money Stock', 'cyan'),
            self.create_plot('Reverse Repo (ON RRP)', 'Reverse Repo', 'orange'),
            self.create_plot('Reserve Balances with Federal Reserve Banks', 'Reservas Bancarias', 'lightgreen'),
            self.create_plot('Velocity of M2 Money Stock', 'Velocidad de M2', '#3B3256'),
        ]


class Nervioso(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "NFCI": "Chicago Fed Financial Conditions Index",
            "VIXCLS": "VIX",
            "BAMLH0A0HYM2": "High Yield Spread",
        }

    def plots(self):
        return [
            self.create_plot('Chicago Fed Financial Conditions Index', 'NFCI - Condiciones Financieras', 'orange'),
            self.create_plot('VIX', 'VIX - Volatilidad Esperada', 'violet'),
            self.create_plot('High Yield Spread', 'High Yield Spread', 'salmon'),
        ]


class Pulmones(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "DGS2": "2Y Treasury Yield",
            "DGS10": "10Y Treasury Yield",
            "BAA10Y": "BAA Spread over 10Y Treasury",
            "TOTBKCR": "Total Bank Credit",
        }

    def prepare(self):
        df = self.data
        df['2Y Treasury Yield - 10Y Treasury Yield'] = df['2Y Treasury Yield'] - df['10Y Treasury Yield']

    def plots(self):
        return [
            self.create_plot('2Y Treasury Yield - 10Y Treasury Yield', 'Curva de Rendimientos', 'red'),
            self.create_plot('BAA Spread over 10Y Treasury', 'BAA Spread', 'violet'),
            self.create_plot('Total Bank Credit', 'Credito bancario total', 'green'),
        ]


class Metabolismo(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "ICSA": "Initial Jobless Claims",
            "PCE": "Personal Consumption Expenditures",
            "AWHAETP": "Avg Weekly Hours (Private Sector)",
            "MANEMP": "All Employees, Manufacturing",
        }

    def plots(self):
        return [
            self.create_plot('Initial Jobless Claims', 'Solicitud seguro desempleo', '#845A5A'),
            self.create_plot('Personal Consumption Expenditures', 'Consumo', '#555C78'),
            self.create_plot('Avg Weekly Hours (Private Sector)', 'Promedio horas trabajadas', '#B6A268'),
            self.create_plot('All Employees, Manufacturing', 'Manufacturing Employment', '#67642D'),
        ]


class Inmunologico(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "COMPOUT": "Commercial Paper Outstanding",
            "BAMLH0A0HYM2": "High Yield Spread",
            "DRTSCILM": "Loan Officer Survey: % Banks Tightening C&I Loans (Large Firms)",
            "WALCL": "Fed Balance Sheet Total Assets",
            "RIFSPPFAAD90NB": "Net Assets of Money Market Funds (AUM)",
            "RRPONTSYD": "Reverse Repo (ON RRP)",
            "H8B1058NCBCAG": "Consumer Loans by Finance Companies",
            "NONREVSL": "Nonrevolving Consumer Credit",
            "BTFPAMOUNTS": "Bank Term Funding Program Usage",
            "REVOLSL": "Revolving Consumer Credit",
            "DRCCLACBS": "Delinquency Rate on Credit Card Loans",
            "DRSFRMACBS": "Delinquency Rate on Single-Family Mortgages",
        }

    def plots(self):
        return [
            self.create_plot('Commercial Paper Outstanding', 'Credito fuera del sistema', '#60695F'),
            self.create_plot('High Yield Spread', 'BBB-AAA Spread', '#3A1010'),
            self.create_plot('Loan Officer Survey: % Banks Tightening C&I Loans (Large Firms)', 'Loan Officer Survey', '#546086'),
            self.create_plot('Fed Balance Sheet Total Assets', 'Fed Balance Sheet Total Assets', '#714955'),
            self.create_plot('Net Assets of Money Market Funds (AUM)', 'Money Market AUM', '#8D8758'),
            self.create_plot('Reverse Repo (ON RRP)', 'Reverse Repo', 'orange'),
            self.create_plot('Consumer Loans by Finance Companies', 'Consumer Loans by Finance Companies', '#373C5F'),
            self.create_plot('Nonrevolving Consumer Credit', 'Nonrevolving Consumer Credit', '#5D6531'),
            self.create_plot('Bank Term Funding Program Usage', 'BTFP Usage', '#FF8F00'),
            self.create_plot('Revolving Consumer Credit', 'Revolving Consumer Credit', '#AB47BC'),
            self.create_plot('Delinquency Rate on Credit Card Loans', 'Delinquency Rate - Credit Card', '#EF5350'),
            self.create_plot('Delinquency Rate on Single-Family Mortgages', 'Delinquency Rate - Mortgages', '#5C6BC0'),
        ]


class Musculatura(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "INDPRO": "Industrial Production Index",
            "TCU": "Capacity Utilization: Total Industry",
            "AMTMNO": "Manufacturers' New Orders: Total Manufacturing",
            "IPMAN": "Industrial Production: Manufacturing",
            "IPB50001N": "Industrial Production: Durable Consumer Goods",
            "IPG331S": "Industrial Production: Mining",
            "CMRMTSPL": "Real Manufacturing and Trade Sales",
            "BUSINV": "Total Business Inventories",
        }

    def plots(self):
        return [
            self.create_plot('Industrial Production Index', 'Producción Industrial Total', '#4CAF50'),
            self.create_plot('Capacity Utilization: Total Industry', 'Utilización de Capacidad', '#FFC107'),
            self.create_plot("Manufacturers' New Orders: Total Manufacturing", "Manufacturers' New Orders", '#2196F3'),
            self.create_plot('Industrial Production: Manufacturing', 'Manufactura', '#E91E63'),
            self.create_plot('Industrial Production: Durable Consumer Goods', 'Bienes Duraderos', '#9C27B0'),
            self.create_plot('Industrial Production: Mining', 'Minería', '#00BCD4'),
            self.create_plot('Real Manufacturing and Trade Sales', 'Real Manufacturing and Trade Sales', '#72614F'),
            self.create_plot('Total Business Inventories', 'Total Business Inventories', '#4A2D2D'),
        ]


class CadenasLogisticas(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "TSIFRGHT": "Freight Transportation Services Index",
            "WPU301": "Truck Transportation PPI",
            "ISRATIO": "Total Business: Inventories to Sales Ratio",
        }

    def plots(self):
        return [
            self.create_plot('Freight Transportation Services Index', 'Freight Transportation Services Index', '#F44336'),
            self.create_plot('Truck Transportation PPI', 'Truck Transportation PPI', '#9C27B0'),
            self.create_plot('Total Business: Inventories to Sales Ratio', 'Inventories to Sales Ratio', '#4CAF50'),
        ]


class Higado(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "NONREVSL": "Nonrevolving Consumer Credit",
            "H8B1058NCBCAG": "Consumer Loans by Finance Companies",
            "BTFPAMOUNTS": "Bank Term Funding Program Usage",
            "REVOLSL": "Revolving Consumer Credit",
            "DRCCLACBS": "Delinquency Rate on Credit Card Loans",
            "DRSFRMACBS": "Delinquency Rate on Single-Family Mortgages",
        }

    def plots(self):
        return [
            self.create_plot('Nonrevolving Consumer Credit', 'Nonrevolving Consumer Credit', '#8BC34A'),
            self.create_plot('Consumer Loans by Finance Companies', 'Consumer Loans by Finance Companies', '#00ACC1'),
            self.create_plot('Bank Term Funding Program Usage', 'Bank Term Funding Program Usage', '#FF8F00'),
            self.create_plot('Revolving Consumer Credit', 'Revolving Consumer Credit', '#AB47BC'),
            self.create_plot('Delinquency Rate on Credit Card Loans', 'Delinquency Rate - Credit Card Loans', '#EF5350'),
            self.create_plot('Delinquency Rate on Single-Family Mortgages', 'Delinquency Rate - Mortgages', '#5C6BC0'),
        ]


class Utero(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "IPX5VHT2N": "Industrial Production: Total Excluding Selected High-Tech and Motor Vehicles & Parts",
            "PNFIC1": "Private Nonresidential Fixed Investment",
            "IPBUSEQ": "Industrial Production: Business Equipment",
            "IPG3254S": "Industrial Production: Pharmaceuticals and Medicine",
            "IPUEN334413T011000000": "Real Sectoral Output for Manufacturing: Semiconductor and Related Device Manufacturing",
            "IPG3341S": "Industrial Production: Computers and Peripheral Equipment",
            "CES6562440001": "All Employees: Scientific Research and Development Services",
            "HOUST5F": "New Privately-Owned Housing Units Started: 5 Units or More",
        }

    def plots(self):
        return [
            self.create_plot('Industrial Production: Total Excluding Selected High-Tech and Motor Vehicles & Parts', 'Producción Industrial excl. High-Tech y Autos', '#009688'),
            self.create_plot('Private Nonresidential Fixed Investment', 'Inversión Fija No Residencial', '#3F51B5'),
            self.create_plot('Industrial Production: Business Equipment', 'Producción de Bienes de Capital', '#FF5722'),
            self.create_plot('Industrial Production: Pharmaceuticals and Medicine', 'Producción Farmacéutica', '#8BC34A'),
            self.create_plot('Real Sectoral Output for Manufacturing: Semiconductor and Related Device Manufacturing', 'Producción de Semiconductores', '#E91E63'),
            self.create_plot('Industrial Production: Computers and Peripheral Equipment', 'Producción de Computadores', '#607D8B'),
            self.create_plot('All Employees: Scientific Research and Development Services', 'Empleo en I+D', '#FFC107'),
            self.create_plot('New Privately-Owned Housing Units Started: 5 Units or More', 'Innovación en Viviendas Multifamiliares', '#9C27B0'),
        ]


class SistemaOseo(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "GDPC1": "Real Gross Domestic Product",
            "A191RL1Q225SBEA": "Real GDP per Capita",
            "GPDIC1": "Gross Private Domestic Investment",
            "TCU": "Capacity Utilization: Total Industry",
            "GDPPOT": "Real Potential Gross Domestic Product",
            "GCEC1": "Real Government Consumption Expenditures and Gross Investment",
            "NETEXP": "Net Exports of Goods and Services",
            "FYFSGDA188S": "Federal Debt to GDP Ratio",
            "PSAVERT": "Personal Saving Rate",
        }

    def plots(self):
        return [
            self.create_plot('Real Gross Domestic Product', 'Real Gross Domestic Product', '#2196F3'),
            self.create_plot('Real GDP per Capita', 'Real GDP per Capita', '#4CAF50'),
            self.create_plot('Gross Private Domestic Investment', 'Gross Private Domestic Investment', '#FFC107'),
            self.create_plot('Capacity Utilization: Total Industry', 'Capacity Utilization', '#E91E63'),
            self.create_plot('Real Potential Gross Domestic Product', 'Real Potential GDP', '#FF5722'),
            self.create_plot('Real Government Consumption Expenditures and Gross Investment', 'Government Investment', '#9C27B0'),
            self.create_plot('Net Exports of Goods and Services', 'Net Exports', '#00BCD4'),
            self.create_plot('Federal Debt to GDP Ratio', 'Federal Debt to GDP Ratio', '#8BC34A'),
            self.create_plot('Personal Saving Rate', 'Personal Saving Rate', '#795548'),
        ]


class Comunicacion(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "UMCSENT": "Consumer Sentiment (UMich)",
            "T5YIFR": "5Y5Y Forward Inflation Expectation",
            "T10YIE": "10Y Breakeven Inflation",
            "VIXCLS": "VIX",
            "EXPINF1YR": "Expected Inflation 1Y Ahead",
        }

    def plots(self):
        return [
            self.create_plot('Consumer Sentiment (UMich)', 'Consumer Sentiment (UMich)', '#42A5F5'),
            self.create_plot('5Y5Y Forward Inflation Expectation', '5Y5Y Forward Inflation Expectation', '#66BB6A'),
            self.create_plot('10Y Breakeven Inflation', '10Y Breakeven Inflation', '#FFA726'),
            self.create_plot('VIX', 'VIX', '#431818'),
            self.create_plot('Expected Inflation 1Y Ahead', 'Expected Inflation 1Y Ahead', '#EF5350'),
        ]


class SistemaAutonomo(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "GCEC1": "Real Government Consumption & Gross Investment",
            "GFDEBTN": "Federal Debt: Total Public Debt",
            "GFDEGDQ188S": "Federal Debt: % of GDP",
            "A091RC1Q027SBEA": "Interest Payments by the Federal Government",
            "MTSDS133FMS": "Federal Surplus or Deficit",
        }

    def plots(self):
        return [
            self.create_plot('Real Government Consumption & Gross Investment', 'Gasto e Inversión Pública (Real)', '#4CAF50'),
            self.create_plot('Federal Debt: Total Public Debt', 'Deuda Pública Total', '#F44336'),
            self.create_plot('Federal Debt: % of GDP', 'Deuda Pública como % del PIB', '#2196F3'),
            self.create_plot('Interest Payments by the Federal Government', 'Pagos de Interés del Gobierno Federal', '#FFC107'),
            self.create_plot('Federal Surplus or Deficit', 'Superávit o Déficit Fiscal', '#9C27B0'),
        ]


class SistemaDigestivo(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "PCE": "Personal Consumption Expenditures",
            "PCECC96": "Real Personal Consumption Expenditures",
            "REVOLSL": "Consumer Credit: Credit Cards and Other Revolving Plans",
            "NONREVSL": "Consumer Credit: Nonrevolving",
        }

    def plots(self):
        return [
            self.create_plot('Personal Consumption Expenditures', 'Personal Consumption Expenditures', '#29B6F6'),
            self.create_plot('Real Personal Consumption Expenditures', 'Real Personal Consumption Expenditures', '#66BB6A'),
            self.create_plot('Consumer Credit: Credit Cards and Other Revolving Plans', 'Revolving Consumer Credit (Credit Cards)', '#FF7043'),
            self.create_plot('Consumer Credit: Nonrevolving', 'Nonrevolving Consumer Credit', '#AB47BC'),
        ]


class Temperatura(Sistema):
    def __init__(self, start="2021-01-01", end=None):
        super().__init__(start, end)
        self.series = {
            "PCEPI": "PCE: Chain-type Price Index",
            "WPUFD49207": "PPI: Intermediate Demand by Production Flow",
            "WPSFD49207": "PPI: Final Demand by Production Flow",
            "GFDEBTN": "Federal Debt: Total Public Debt",
        }

    def plots(self):
        return [
            self.create_plot('PCE: Chain-type Price Index', 'PCE: Chain-type Price Index', '#F44336'),
            self.create_plot('PPI: Intermediate Demand by Production Flow', 'PPI - Intermediate Demand', '#FF9800'),
            self.create_plot('PPI: Final Demand by Production Flow', 'PPI - Final Demand', '#FFC107'),
            self.create_plot('Federal Debt: Total Public Debt', 'Total Public Debt', '#D32F2F'),
        ]


class AppInterface:
    def __init__(self):
        self.sistemas = {
            "🫀 Circulatorio": Circulatorio(),
            "🧠 Nervioso": Nervioso(),
            "🫁 Pulmones": Pulmones(),
            "🧬 Metabolismo": Metabolismo(),
            "🧪 Inmunológico (Shadow banking)": Inmunologico(),
            "Musculatura (Produccion industrial)": Musculatura(),
            "Cadenas logisticas": CadenasLogisticas(),
            "Higado (Sistema bancario)": Higado(),
            "Utero (Innovacion y desarollo)": Utero(),
            "Sistema Oseo(Estructura economica)": SistemaOseo(),
            "Comunicacion(Sentimiento de mercado)": Comunicacion(),
            "Sistema autonomo(Politica fiscal y monetaria)": SistemaAutonomo(),
            "Sistema digestivo(consumo)": SistemaDigestivo(),
            "Temperatura(Inflacion)": Temperatura(),
        }

    def run(self):
        st.set_page_config(layout="wide")
        st.title("🧍 Anatomía Económica del Mercado")
        sistema = st.sidebar.selectbox("Selecciona un sistema", list(self.sistemas.keys()))
        seleccionado = self.sistemas[sistema]
        for fig in seleccionado.plots():
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    AppInterface().run()
