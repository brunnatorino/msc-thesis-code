
import pandas as pd
from pathlib import Path
import yfinance as yf


class DataPaths:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.bonds = self.base / "Desktop/Thesis/data/bonds/model"
        self.macro = self.base / "Desktop/Thesis/data/macro"
        self.yf = self.macro / "yf"


class FXDataManager:
    def __init__(self, paths: DataPaths):
        self.paths = paths

    def load_currency(self, symbol: str, name: str, invert: bool = False) -> pd.DataFrame:
        df = self.get_yf_data(symbol)[['Date', 'Close']].rename(columns={'Date': 'DATE'})
        df['DATE'] = df['DATE'].str[:10]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = self.normalize_periods(df)
        df.columns = [name]
        if invert:
            df[name] = 1 / df[name]
        return df

    def get_combined_fx(self) -> pd.DataFrame:
        eur = self.load_currency('EURUSD', 'eur')
        brl = self.load_currency('BRLUSD', 'brl')
        clp = self.load_currency('CLPUSD', 'clp', invert=True)
        return pd.concat([eur, brl, clp], axis=1)

    def get_yf_data(self, ticker: str) -> pd.DataFrame:
        return pd.read_csv(self.paths.yf / f"{ticker}.csv")

    def convert_to_usd(self, df: pd.DataFrame, col: str, currency: str) -> pd.DataFrame:
        fx_dict = self.get_combined_fx().resample('MS').mean().to_dict()
        df['fx'] = df.index.map(fx_dict[currency])
        df[col] *= df['fx']
        return df.drop(columns='fx')

    @staticmethod
    def normalize_periods(df: pd.DataFrame) -> pd.DataFrame:
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df.set_index('DATE').resample('MS').mean()


class BondDataManager:
    def __init__(self, paths: DataPaths):
        self.paths = paths

    def load_sector_spreads(self, tenors: list[int]) -> dict[int, pd.DataFrame]:
        spreads = {}
        for tenor in tenors:
            path = self.paths.bonds / 'IT_BR' / f'monthly_spreads_{tenor}.csv'
            df = pd.read_csv(path).rename(columns={'date': 'DATE'})
            df = self.normalize_periods(df)
            spreads[tenor] = df
        return spreads

    @staticmethod
    def normalize_periods(df: pd.DataFrame) -> pd.DataFrame:
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df.set_index('DATE').resample('MS').mean()

