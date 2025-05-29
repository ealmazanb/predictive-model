import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential, Input
from keras.layers import LSTM, Dense, Dropout

'''
Config params
'''
EPOCH = 6
BATCH_SIZE = 16
LOOKBACK_DAYS = 30

class LSTMForecastModel:
    def __init__(self, lookback=LOOKBACK_DAYS, loss_function='mse', use_logs=True):
        self.lookback = lookback
        self.loss_function = loss_function
        self.use_logs = use_logs

        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.target_col = None

    def _prepare_xy(self, df: pd.DataFrame):
        data = df[self.feature_cols + [self.target_col]].values
        scaled_data = self.scaler.fit_transform(data)

        target_index = len(self.feature_cols)

        X, y = [], []
        for i in range(len(scaled_data) - self.lookback - 1):
            X.append(scaled_data[i:i + self.lookback, :len(self.feature_cols)])
            y.append(scaled_data[i + self.lookback, target_index])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss=self.loss_function)
        return model

    def train(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        df = df.copy().dropna(subset=feature_cols)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = MinMaxScaler()
        X, y = self._prepare_xy(df)
        self.model = self._build_model((X.shape[1], X.shape[2]))
        self._log(f"[LSTM] Training on {X.shape[0]}")
        self.model.fit(X, y, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=0)
        self._log("[LSTM] Training complete.")

    def predict(self, df: pd.DataFrame):
        df = df.copy().dropna(subset=self.feature_cols + [self.target_col])
        recent = df.tail(self.lookback)[self.feature_cols].values
        scaled_input = self.scaler.transform(np.hstack([recent, np.zeros((self.lookback, 1))]))
        x_input = scaled_input[:, :len(self.feature_cols)].reshape(1, self.lookback, len(self.feature_cols))

        y_pred_scaled = self.model.predict(x_input, verbose=0)[0][0]

        dummy_row = np.zeros(len(self.feature_cols) + 1)
        dummy_row[-1] = y_pred_scaled
        inv_value = self.scaler.inverse_transform([dummy_row])[0][-1]
        self._log(f"[LSTM] Predicted next value: {inv_value:.4f}")
        return inv_value

    def _log(self, text):
        if self.use_logs:
            print(text)
