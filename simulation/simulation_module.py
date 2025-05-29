import pandas as pd
from datetime import timedelta

from decision import DecisionManager
from prediction import LSTMForecastModel
from prediction.random_prediction_module import RandomPredictiveModel
from prediction.shift_supervised_prediction_module import ShiftPredictiveModel
from utils.utils_module import Utils
from prediction.arima_prediction_module import TimeSeriesPredictiveModel


class Simulator:
    def __init__(self, simulation_list, df, initial_liquidity, algorithm_type,
                 sl_min, sl_max, tp_min, tp_max,
                 reserve=0.1, operate_in_weekends=False, use_logs=True):
        self.direction_total = 0
        self.direction_hits = 0
        self.assets = simulation_list
        self.df = df.set_index('timestamp')
        self.liquidity = initial_liquidity
        self.reserve = reserve
        self.asset_value = 0
        self.algorithm_type = algorithm_type
        self.operate_in_weekends = operate_in_weekends
        self.use_logs = use_logs
        self.tp_min = tp_min
        self.tp_max = tp_max
        self.sl_min = sl_min
        self.sl_max = sl_max
        self.portfolio = {}
        self.transactions = []
        self.model = self.get_model()
        self.decision_manager = DecisionManager(
            sl_min=sl_min,
            sl_max=sl_max,
            tp_min=tp_min,
            tp_max=tp_max
        )

    def get_model(self):
        if self.algorithm_type == 'ARIMA':
            return TimeSeriesPredictiveModel()
        elif self.algorithm_type == 'RANDOM':
            return RandomPredictiveModel()
        elif self.algorithm_type == 'SHIFT':
            return ShiftPredictiveModel()
        elif self.algorithm_type == 'LSTM':
            return LSTMForecastModel()
        raise NotImplementedError("Modelo no implementado")

    def run(self, start_date, end_date):
        utils = Utils()
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        print(f"Starting simulation from {start_date} to {end_date}")

        while current_date <= end_date:
            if not self.operate_in_weekends and utils.is_weekend(current_date):
                current_date += timedelta(days=1)
                continue

            for asset in self.assets:
                self._simulate_asset(asset, current_date)

            current_date += timedelta(days=1)
        if self.direction_total > 0:
            accuracy = (self.direction_hits / self.direction_total) * 100
            print(f"Directional accuracy: {accuracy:.2f}% ({self.direction_hits}/{self.direction_total})")

    def _simulate_asset(self, asset, date):
        utils = Utils()

        past_data = self.df.loc[self.df.index < date].reset_index()
        if len(past_data) < 40:
            return

        try:
            print(f"[{date.date()}] Training {asset}")
            if self.algorithm_type == 'SHIFT':
                self.model.train(asset, past_data)
                predicted_price = self.model.predict(asset, past_data)

            elif self.algorithm_type == 'LSTM':
                target_col = f"{asset}_value"
                feature_cols = [
                    col for col in past_data.columns
                    if col.startswith(f"{asset}_feature_")
                       or col == f"{asset}_sentiment"
                       or col == f"{asset}_volume"
                ]
                macro_cols = [col for col in past_data.columns if col.startswith("usa_") or col.startswith("euro_")]
                feature_cols += macro_cols
                self.model.train(past_data, feature_cols=feature_cols, target_col=target_col)
                predicted_price = self.model.predict(past_data)

            else:
                self.model.train(past_data, target_col=f"{asset}_feature_ma_10")
                forecast = self.model.predict(horizon=1)
                predicted_price = forecast.iloc[0]

            print(f"[{date.date()}] Forecast for {asset}: {predicted_price:.2f}")
        except Exception as e:
            if self.use_logs:
                print(f"[{date.date()}] Prediction error for {asset}: {e}")
            return

        real_price = utils.get_price(self.df, date, asset)
        yesterday = date - timedelta(days=1)
        yesterday_price = utils.get_price(self.df, yesterday, asset)
        if yesterday_price is not None:
            real_diff = real_price - yesterday_price
            predicted_diff = predicted_price - yesterday_price
            if (real_diff * predicted_diff) > 0:
                self.direction_hits += 1
            self.direction_total += 1
        if real_price is None:
            print(f"[{date.date()}] No price data for {asset}")
            return

        self._apply_decision_logic(asset, predicted_price, real_price, date)

    def _apply_decision_logic(self, asset, predicted, current, date):
        action = self.decision_manager.decide_action(
            asset=asset,
            predicted=predicted,
            current=current,
            liquidity=self.liquidity,
            portfolio=self.portfolio
        )
        print(
            f"[{date.date()}] {asset} | Current: {current:.2f} | Predicted: {predicted:.2f} | Action: {action['type']}")

        if action['type'] is None or action['quantity'] <= 0:
            return

        self._execute_transaction(
            asset=asset,
            price=action['price'],
            qty=action['quantity'],
            op_type=action['type'],
            date=date
        )

    def _execute_transaction(self, asset, price, qty, op_type, date):
        if op_type == 'buy':
            total_cost = price * qty
            self.liquidity -= total_cost

            if asset not in self.portfolio:
                self.portfolio[asset] = {'quantity': qty, 'avg_price': price}
            else:
                p = self.portfolio[asset]
                total_qty = p['quantity'] + qty
                avg_price = ((p['quantity'] * p['avg_price']) + (qty * price)) / total_qty
                self.portfolio[asset] = {'quantity': total_qty, 'avg_price': avg_price}

            if self.use_logs:
                print(f"[{date.date()}] BUY {asset} | Qty: {qty} | Price: {price:.2f}")

        elif op_type == 'sell':
            if asset not in self.portfolio:
                return

            avg_price = self.portfolio[asset]['avg_price']
            profit_pct = ((price - avg_price) / avg_price) * 100
            self.liquidity += price * qty
            del self.portfolio[asset]

            if self.use_logs:
                print(f"[{date.date()}] SELL {asset} | Qty: {qty} | Price: {price:.2f} | Gain: {profit_pct:.2f}%")
        self.transactions.append({
            'timestamp': date,
            'code': asset,
            'price': price,
            'quantity': qty,
            'type': op_type
        })
