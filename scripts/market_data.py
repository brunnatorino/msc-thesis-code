import pandas as pd
import numpy as np
from arch.univariate import ARX, ARCH
from pathlib import Path

class MarketDataManager:
    def __init__(self, get_yf_func, normalize_func, local_drive: Path, databonds: Path):
        self.get_yf = get_yf_func
        self.normalize = normalize_func
        self.local_drive = local_drive
        self.databonds = databonds

    def _load_yf_series(self, ticker: str, column_name: str) -> pd.DataFrame:
        df = self.get_yf(ticker)[['Date', 'Close']].rename(columns={'Date': 'DATE'})
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['DATE'] = df['DATE'].astype(str).str[:10]
        df = self.normalize(df)
        df.columns = [column_name]
        return df

    def get_vix(self) -> pd.DataFrame:
        return self._load_yf_series('VIX', 'vix')

    def get_us3m(self) -> pd.DataFrame:
        return self._load_yf_series('DTB3', 'us3m')

    def get_gold(self) -> pd.DataFrame:
        return self._load_yf_series('GOLD', 'gold')

    def get_commods(self) -> pd.DataFrame:
        return self._load_yf_series('DBC', 'commods')

    def get_us_high_yield(self) -> pd.DataFrame:
        df = self.get_yf('US_HIGH_YIELD')[['DATE', 'BAMLH0A0HYM2EY']]
        df.rename(columns={'BAMLH0A0HYM2EY': 'us_high_yield'}, inplace=True)
        df['us_high_yield'] = pd.to_numeric(df['us_high_yield'], errors='coerce')
        df.dropna(inplace=True)
        df['DATE'] = df['DATE'].astype(str).str[:10]
        df = self.normalize(df)
        df.columns = ['us_high_yield']
        return df

    def get_fx(self) -> pd.DataFrame:
        eur = self._load_yf_series('EURUSD', 'eur')
        brl = self._load_yf_series('BRLUSD', 'brl')
        clp = self._load_yf_series('CLPUSD', 'clp')
        return pd.concat([eur, brl, clp], axis=1)

    def get_contagion(self) -> pd.DataFrame:
        itvix = pd.read_csv(self.local_drive / self.databonds / 'itvix.csv')[['Date', 'Price']]
        brvix = pd.read_csv(self.local_drive / self.databonds / 'brvix.csv')[['Date', 'Price']]

        itvix.rename(columns={'Date': 'DATE'}, inplace=True)
        brvix.rename(columns={'Date': 'DATE'}, inplace=True)

        itvix['Price'] = pd.to_numeric(itvix['Price'].astype(str).str.replace(',', ''))
        brvix['Price'] = pd.to_numeric(brvix['Price'].astype(str).str.replace(',', ''))

        itvix = self.normalize(itvix)
        brvix = self.normalize(brvix)

        itvix.columns = ['it_contagion']
        brvix.columns = ['br_contagion']

        ar = ARX(brvix, lags=[3])
        ar.volatility = ARCH(p=5)
        res = ar.fit(update_freq=0, disp="off")
        vol = pd.DataFrame(res.conditional_volatility.dropna(), index=brvix.index[-len(res.conditional_volatility):])
        vol.columns = ['br_contagion']
        brvix['br_contagion'] = vol['br_contagion']

        contagion = pd.concat([itvix, brvix], axis=1)
        return contagion.dropna()

