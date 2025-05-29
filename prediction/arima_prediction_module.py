from datetime import timedelta
import pandas as pd
from pmdarima import auto_arima


class TimeSeriesPredictiveModel:

    def __init__(self, use_logs=True, seasonal=False, max_order=(6, 6, 6), window_size=30):
        self.use_logs = use_logs
        self.seasonal = seasonal
        self.max_order = max_order
        self.model = None
        self.fitted_model = None
        self.last_train_index = None
        self.window_size = window_size

    def train(self, df: pd.DataFrame, target_col: str):
        ts = df.set_index('timestamp')[target_col].dropna().asfreq('D')
        self.last_train_index = ts.index[-1]
        self.model = auto_arima(
            ts[-self.window_size:],
            seasonal=self.seasonal,
            start_p=0,
            start_q=0,
            max_p=self.max_order[0],
            max_d=self.max_order[1],
            max_q=self.max_order[2],
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        self._log(f"[arima] Model trained with order: {self.model.order}")
        self._log(f"[arima] Last training date: {self.last_train_index.date()}")

    def predict(self, horizon=1):
        forecast = self.model.predict(n_periods=horizon)
        dates = pd.date_range(start=self.last_train_index + timedelta(days=1), periods=horizon)
        self._log(f"[arima] Prediction from {dates[0].date()} to {dates[-1].date()}")
        return pd.Series(forecast, index=dates, name='forecast')

    def _log(self, msg):
        if self.use_logs:
            print(msg)