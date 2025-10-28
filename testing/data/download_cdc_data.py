import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def download_cdc_fluview_data(output_path: str = "cdc_flu_data.csv") -> pd.DataFrame:
    print("Downloading CDC data via Delphi API...")
    base_url = "https://api.delphi.cmu.edu/epidata/fluview/"

    epiweeks = []
    for year in range(2010, 2024):
        epiweeks.extend([f"{year}{w:02d}" for w in range(40, 53)])
        epiweeks.extend([f"{year+1}{w:02d}" for w in range(1, 21)])

    params = {"regions": ",".join([f"hhs{i}" for i in range(1, 11)]), "epiweeks": ",".join(epiweeks)}

    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get("result") != 1:
            raise ValueError(f"API error: {data.get('message')}")

        df = pd.DataFrame(data.get("epidata", []))
        df = df.rename(columns={'region': 'REGION', 'epiweek': 'EPIWEEK', 'wili': '%WEIGHTED ILI',
                                'ili': 'ILI', 'num_providers': 'NUM. OF PROVIDERS', 'total_patients': 'TOTAL PATIENTS'})
        df['YEAR'] = df['EPIWEEK'] // 100
        df['WEEK'] = df['EPIWEEK'] % 100
        df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + df['WEEK'].astype(str).str.zfill(2) + '1', format='%Y%W%w')
        df['REGION'] = df['REGION'].str.replace('hhs', 'Region ')
        df.to_csv(output_path, index=False)
        print(f"Downloaded {len(df)} rows of real CDC data")
        return df
    except Exception as e:
        print(f"API failed: {e}, using simulated data")
        return create_simulated_cdc_data('cdc_flu_data_fake.csv')


def create_simulated_cdc_data(output_path: str = "cdc_flu_data.csv") -> pd.DataFrame:
    print("Generating simulated CDC data...")
    records = []
    start_date = datetime(2013, 10, 1)
    regions = [f"Region {i}" for i in range(1, 11)]

    for season in range(10):
        season_start = start_date + timedelta(weeks=52 * season)
        for week in range(52):
            week_date = season_start + timedelta(weeks=week)
            for region in regions:
                np.random.seed(season * 1000 + week * 10 + regions.index(region))
                week_of_season = week % 52
                seasonal = np.exp(-((week_of_season - 10) ** 2) / (2 * 8 ** 2))
                ili_rate = 1.0 + 4.0 * seasonal * np.random.uniform(0.5, 2.0) + np.random.normal(0, 0.3)
                ili_rate = max(0.1, ili_rate)
                total = np.random.randint(50000, 150000)
                records.append({
                    'REGION': region, 'YEAR': week_date.year, 'WEEK': week_date.isocalendar()[1],
                    'DATE': week_date.strftime('%Y-%m-%d'), '%WEIGHTED ILI': ili_rate,
                    'ILI': int(ili_rate * total / 100), 'TOTAL PATIENTS': total,
                    'NUM. OF PROVIDERS': np.random.randint(100, 500)
                })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows")
    return df


def generate_synthetic_outbreak_data(output_path: str = "synthetic_flu_data.csv") -> pd.DataFrame:
    print("Generating synthetic outbreak data...")
    np.random.seed(42)
    records = []

    for season in range(15):
        season_start = datetime(2008 + season, 10, 1)
        from scipy.stats import genextreme
        peak = max(1000, genextreme.rvs(c=-0.15, loc=5000, scale=1500, size=1)[0])
        peak_week = np.random.randint(8, 16)

        for week in range(40):
            week_date = season_start + timedelta(weeks=week)
            cases = peak * np.exp(-(abs(week - peak_week) ** 2) / (2 * 4 ** 2))
            cases += np.random.normal(0, cases * 0.1)
            cases = max(100, cases)
            records.append({
                'season': f"{2008 + season}-{2009 + season}", 'week': week,
                'date': week_date.strftime('%Y-%m-%d'), 'cases': int(cases), 'season_id': season
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows")
    return df


if __name__ == "__main__":
    download_cdc_fluview_data("cdc_flu_data.csv")
    generate_synthetic_outbreak_data("synthetic_flu_data.csv")
    print("\n" + "="*80)
    print("Data ready!")
    print("="*80)
