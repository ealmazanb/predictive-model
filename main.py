import os

import pandas as pd

from feature import FeatureEngineeringModule
from ingestion import DataIngestionModule
from simulation import Simulator

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_INPUT_DIR = 'input'
SENTIMENT_DIR = 'sentiment'
MACRO_DIR = 'macro'
HISTORIC_DIR = 'historic'
FILES_DIR = 'files'

'''
Config variables
'''
simulation_date_start = '2019-01-01'
simulation_date_end = '2019-01-30'
liquidity = 100_000
reserve = 0.1
tp_min = 0.01
tp_max = 0.05
sl_min = 0.0
sl_max = 0.05  # If 1.0, it will never sell on losses
algorithm_type = 'LSTM'

'''
Data ingestion
'''
sentiment_files = {
    'alibaba': 'alibaba.csv',
    'apple': 'apple.csv',
    'astrazeneca': 'astrazeneca.csv',
    'bitcoin': 'bitcoin.csv',
    'boeing': 'boeing.csv',
    'coca_cola': 'coca_cola.csv',
    'ethereum': 'ethereum.csv',
    'eu_bonds': 'eu_bonds.csv',
    'gold': 'gold.csv',
    'intel': 'intel.csv',
    'johnson_johnson': 'johnson_johnson.csv',
    'lockheed_martin': 'lockheed_martin.csv',
    'oil': 'oil.csv',
    'petrobras': 'petrobras.csv',
    'tesla': 'tesla.csv',
    'us_bonds': 'us_bonds.csv',
}

macro_files = {
    'usa_gdp': 'usa_gdp.csv',
    'usa_inflation': 'usa_inflation.csv',
    'usa_unemployment': 'usa_unemployment.csv',
    'usa_interest_rate': 'usa_interest_rate.csv',
    'euro_gdp': 'euro_gdp.csv',
    'euro_inflation': 'euro_inflation.csv',
    'euro_unemployment': 'euro_unemployment.csv',
    'euro_interest_rate': 'euro_interest_rate.csv',
}

historical_asset_files = {
    'alibaba': 'alibaba.csv',
    'apple': 'apple.csv',
    'astrazeneca': 'astrazeneca.csv',
    'bitcoin': 'bitcoin.csv',
    'boeing': 'boeing.csv',
    'coca_cola': 'coca_cola.csv',
    'ethereum': 'ethereum.csv',
    'eu_bonds': 'eu_bonds.csv',
    'gold': 'gold.csv',
    'intel': 'intel.csv',
    'johnson_johnson': 'johnson_johnson.csv',
    'lockheed_martin': 'lockheed_martin.csv',
    'oil': 'oil.csv',
    'petrobras': 'petrobras.csv',
    'tesla': 'tesla.csv',
    'us_bonds': 'us_bonds.csv',
}

assets = historical_asset_files.keys()  # Using all assets
sentiment_paths = {k: os.path.join(SENTIMENT_DIR, v) for k, v in sentiment_files.items()}
macro_paths = {k: os.path.join(MACRO_DIR, v) for k, v in macro_files.items()}
asset_paths = {k: os.path.join(HISTORIC_DIR, v) for k, v in historical_asset_files.items()}
ingestor = DataIngestionModule(data_dir=BASE_INPUT_DIR)
ingestor.ingest_sentiment_data(sentiment_paths)
ingestor.ingest_macro_data(macro_paths)
ingestor.ingest_asset_data(asset_paths)
df = ingestor.align_all_data()

'''
Part II - Feature engineering
'''
feature_module = FeatureEngineeringModule(df)
feature_module.apply_to_all_assets(assets)
feature_module.scale_standard()
feature_module.get_featured_data()
df = feature_module.remove_na_rows()
df.to_csv(f'{FILES_DIR}/full_dataset.csv', index=False)

'''
Part III - Simulation call
'''
simulator = Simulator(
    simulation_list=list(assets),
    df=df,
    initial_liquidity=liquidity,
    algorithm_type=algorithm_type,
    sl_min=sl_min,
    sl_max=sl_max,
    tp_min=tp_min,
    tp_max=tp_max,
    reserve=reserve,
    operate_in_weekends=False,
    use_logs=True
)

simulator.run(simulation_date_start, simulation_date_end)

transactions_df = pd.DataFrame(simulator.transactions)
transactions_df.to_csv(f"{FILES_DIR}/simulation_auto_arima_20180101_20250101", index=False)

final_value = simulator.liquidity
for asset, data in simulator.portfolio.items():
    current_price = df[df['timestamp'] == simulation_date_end][f"{asset}_value"].values
    if len(current_price) > 0:
        final_value += data['quantity'] * current_price[0]

print(f"Final value: {final_value:,.2f}")
