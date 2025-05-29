import pandas as pd
import os


class DataIngestionModule:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = {}

    def load_csv(self, filename, date_col="timestamp"):
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path, parse_dates=[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)
        return df

    def ingest_asset_data(self, asset_files):
        """
        asset_files: dict with asset_code -> filename
        """
        for asset, filename in asset_files.items():
            df = self.load_csv(filename)
            df = df.rename(columns={"value": f"{asset}_value", "volume": f"{asset}_volume"})
            self.data[asset] = df

    def ingest_sentiment_data(self, sentiment_files):
        """
        sentiment_files: dict with asset_code -> filename
        """
        for asset, filename in sentiment_files.items():
            df = self.load_csv(filename)
            df = df.rename(columns={"sentiment": f"{asset}_sentiment"})
            self.data[f"{asset}_sentiment"] = df

    def ingest_macro_data(self, macro_files):
        """
        macro_files: dict with indicator_name -> filename
        """
        for indicator, filename in macro_files.items():
            df = self.load_csv(filename)
            df = df.rename(columns={"value": indicator})
            self.data[indicator] = df

    def align_all_data(self):
        all_dfs = list(self.data.values())
        base = all_dfs[0][["timestamp"]].copy()

        for df in all_dfs[0:]:
            base = pd.merge(base, df, on="timestamp", how="outer")

        base = base.sort_values("timestamp").reset_index(drop=True)
        base = base.fillna(0)
        self.final_df = base
        return base
