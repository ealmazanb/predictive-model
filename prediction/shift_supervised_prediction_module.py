from sklearn.linear_model import LinearRegression


class ShiftPredictiveModel:
    def __init__(self, shift=1, use_logs=True, base_model_cls=LinearRegression):
        self.shift = shift
        self.use_logs = use_logs
        self.window = None
        self.models = {}
        self.feature_cols = {}
        self.base_model_cls = base_model_cls
        self.target_col_suffix = "_value"

    def train(self, asset, df, window=30):
        self.window = window
        target_col = f"{asset}{self.target_col_suffix}"
        df = df.copy()

        df[f"{asset}_target"] = df[target_col].shift(-self.shift)

        static_features = [
            col for col in df.columns
            if (
                col.startswith(f"{asset}_feature_")
                or col == f"{asset}_sentiment"
                or col == f"{asset}_volume"
                or col.startswith("usa_")
                or col.startswith("euro_")
            )
        ]

        for lag in range(1, window + 1):
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
            static_features.append(f"{target_col}_lag_{lag}")

        df = df.dropna(subset=static_features + [f"{asset}_target"])

        self.feature_cols[asset] = static_features
        x_train = df[static_features]
        y_train = df[f"{asset}_target"]

        model = self.base_model_cls()
        model.fit(x_train, y_train)
        self.models[asset] = model

        self._log(f"[Shift] Trained {asset} with {len(x_train)} samples and {len(static_features)} features")

    def predict(self, asset, df):
        df = df.copy()
        target_col = f"{asset}{self.target_col_suffix}"
        for lag in range(1, self.window + 1):
            lag_col = f"{target_col}_lag_{lag}"
            if lag_col not in df.columns:
                df[lag_col] = df[target_col].shift(lag)

        df = df.dropna(subset=self.feature_cols[asset])
        if df.empty:
            raise ValueError(f"No valid data to predict {asset}")

        latest_features = df[self.feature_cols[asset]].iloc[-1:]
        pred = self.models[asset].predict(latest_features)[0]
        self._log(f"[Shift] Predict {asset}: {pred:.4f}")
        return pred

    def _log(self, msg):
        if self.use_logs:
            print(msg)
