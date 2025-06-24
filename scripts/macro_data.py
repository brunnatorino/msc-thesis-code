# filepath: macro_data_manager.py

import pandas as pd
import numpy as np
from pathlib import Path
from arch.univariate import ARX, ARCH


class MacroDataManager:
    def __init__(self, local_drive: Path, datamacro: Path, normalize_func, diff_func, ratio_func, fx_converter, yf_loader):
        self.local_drive = local_drive
        self.datamacro = datamacro
        self.normalize = normalize_func
        self.calc_diff = diff_func
        self.calc_ratio = ratio_func
        self.convert_to_usd = fx_converter
        self.get_yf = yf_loader

    def get_gdp(self, growth: bool, lag: bool) -> pd.DataFrame:
        gdplevel = pd.read_csv(self.local_drive / self.datamacro / 'model/oecd_weekly_gdp_levels.csv')
        percap = pd.read_csv(self.local_drive / self.datamacro / 'model/oecd_weekly_gdp_level_percapita.csv')

        gdplevel.drop(columns='Unnamed: 0', inplace=True)
        percap.drop(columns='Unnamed: 0', inplace=True)
        gdplevel.rename(columns={'date': 'DATE'}, inplace=True)
        percap.rename(columns={'date': 'DATE'}, inplace=True)

        gdplevel = gdplevel.pivot(index='DATE', columns='region', values='GDP').reset_index()
        gdplevel.columns = ['DATE', 'Brazil_GDP', 'Chile_GDP', 'Italy_GDP', 'United States_GDP']

        percap = percap.pivot(index='DATE', columns='region', values='GDP').reset_index()
        percap.columns = ['DATE', 'Brazil_GDPcapita', 'Chile_GDPcapita', 'Italy_GDPcapita', 'United States_GDPcapita']

        gdplevel['DATE'] = pd.to_datetime(gdplevel['DATE'])
        percap['DATE'] = pd.to_datetime(percap['DATE'])

        gdplevel = self.normalize(gdplevel)
        percap = self.normalize(percap)

        if growth:
            gdplevel = gdplevel.pct_change(periods=12) * 100
            percap = percap.pct_change(periods=12) * 100
            gdplevel.dropna(inplace=True)
            percap.dropna(inplace=True)

        # Calculate differences and ratios
        comparisons = [
            (['Brazil_GDP', 'Italy_GDP'], 'diff_br_it_gdp'),
            (['Brazil_GDPcapita', 'Italy_GDPcapita'], 'diff_br_it_gdp_per_cp'),
            (['Brazil_GDP', 'United States_GDP'], 'diff_br_us_gdp'),
            (['Brazil_GDPcapita', 'United States_GDPcapita'], 'diff_br_us_gdp_per_cp'),
            (['Italy_GDP', 'United States_GDP'], 'diff_it_us_gdp'),
            (['Italy_GDPcapita', 'United States_GDPcapita'], 'diff_it_us_gdp_per_cp'),
            (['Chile_GDP', 'United States_GDP'], 'diff_ch_us_gdp'),
            (['Chile_GDPcapita', 'United States_GDPcapita'], 'diff_ch_us_gdp_per_cp'),
        ]

        for (cols, name) in comparisons:
            if 'capita' in cols[0]:
                percap = self.calc_diff(percap, cols, name)
            else:
                gdplevel = self.calc_diff(gdplevel, cols, name)

        ratios = [
            (['Brazil_GDP', 'United States_GDP'], 'ratio_br_us_gdp'),
            (['Brazil_GDPcapita', 'United States_GDPcapita'], 'ratio_br_us_gdp_per_cp'),
            (['Italy_GDP', 'United States_GDP'], 'ratio_it_us_gdp'),
            (['Italy_GDPcapita', 'United States_GDPcapita'], 'ratio_it_us_gdp_per_cp'),
        ]

        for (cols, name) in ratios:
            if 'capita' in cols[0]:
                percap = self.calc_ratio(percap, cols, name)
            else:
                gdplevel = self.calc_ratio(gdplevel, cols, name)

        gdp = pd.concat([gdplevel, percap], axis=1).dropna()

        if lag:
            gdp = gdp.shift(3).dropna()

        return gdp

    def get_inflation(self) -> pd.DataFrame:
        br = pd.read_csv(self.local_drive / self.datamacro / 'model/br_inflation.csv')
        it = pd.read_csv(self.local_drive / self.datamacro / 'model/it_inflation.csv')

        br['br_inflation_mom'] = br['IPCA'].pct_change() * 100
        br['br_inflation_yoy'] = (br['IPCA'] / br['IPCA'].shift(12) - 1) * 100
        br.drop(columns='IPCA', inplace=True)

        it['it_inflation_mom'] = it['HICP - All Items'].pct_change() * 100
        it['it_inflation_yoy'] = (it['HICP - All Items'] / it['HICP - All Items'].shift(12) - 1) * 100
        it.drop(columns='HICP - All Items', inplace=True)

        it.rename(columns={'Date': 'DATE'}, inplace=True)
        br.dropna(inplace=True)
        it.dropna(inplace=True)

        it['DATE'] = pd.to_datetime(it['DATE'])
        br['DATE'] = pd.to_datetime(br['DATE'])

        it = self.normalize(it)
        br = self.normalize(br)

        df = pd.concat([br, it], axis=1).dropna()
        df = self.calc_diff(df, ['br_inflation_yoy', 'it_inflation_yoy'], 'diff_br_it_inflation_yoy')
        df = self.calc_diff(df, ['br_inflation_mom', 'it_inflation_mom'], 'diff_br_it_inflation_mom')

        return df
