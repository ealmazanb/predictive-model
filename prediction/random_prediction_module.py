import pandas as pd
import numpy as np
from datetime import timedelta


class RandomPredictiveModel:
    def __init__(self, use_logs=True, volatility=0.02, seed=None):
        self.use_logs = use_logs
        self.volatility = volatility
        self.last_train_index = None
        self.current_price = None
        if seed is not None:
            np.random.seed(seed)

    def train(self, df: pd.DataFrame, target_col: str):
        ts = df.set_index('timestamp')[target_col].dropna().asfreq('D')
        self.last_train_index = ts.index[-1]
        self.current_price = ts.iloc[-1]
        self._log(f"[random] Using price {self.current_price:.2f} from {self.last_train_index.date()} as base")

    def predict(self, horizon=1):
        returns = np.random.normal(loc=0, scale=self.volatility, size=horizon)
        forecast = [self.current_price * (1 + r) for r in returns]
        dates = pd.date_range(start=self.last_train_index + timedelta(days=1), periods=horizon)
        self._log(f"[random] Prediction from {dates[0].date()} to {dates[-1].date()}")
        return pd.Series(forecast, index=dates, name='forecast')

    def _log(self, msg):
        if self.use_logs:
            print(msg)
