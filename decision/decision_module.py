import random


class DecisionManager:
    """
    strategies
    - PROPORTIONAL: If buy signal is higher, buy more. Based on alpha variable
    - FIXED_PERCENT: Always buys the same based on fixed_pct variable
    - RANDOM: Buys a random percent
    - FIXED: True fixed
    """

    def __init__(self, tp_min, tp_max, sl_min, sl_max, reserve=0.1, use_logs=True, strategy='PROPORTIONAL', alpha=0.5,
                 fixed_pct=0.1, fixed=10000):
        self.tp_min = tp_min
        self.tp_max = tp_max
        self.sl_min = sl_min
        self.sl_max = sl_max
        self.reserve = reserve
        self.use_logs = use_logs
        self.strategy = strategy
        self.alpha = alpha
        self.fixed_pct = fixed_pct
        self.fixed = fixed

    def decide_action(self, asset, predicted, current, liquidity, portfolio):
        expected_return = (predicted - current) / current
        action = {'type': None, 'quantity': 0, 'price': current}

        if expected_return > self.tp_min:
            max_liquidity = liquidity * (1 - self.reserve)
            quantity = 0
            if self.strategy == 'PROPORTIONAL':
                quantity = int((self.alpha * expected_return) * (max_liquidity // current))
                quantity = max(quantity, 0)
            elif self.strategy == 'FIXED_PERCENT':
                quantity = int((self.fixed_pct * max_liquidity) // current)
            elif self.strategy == 'FIXED':
                quantity = int(self.fixed // current)
            elif self.strategy == 'RANDOM':
                random_pct = random.uniform(0.05, 0.5)
                quantity = int((random_pct * max_liquidity) // current)

            if quantity > 0:
                action.update({'type': 'buy', 'quantity': quantity})
                return action

        if asset in portfolio:
            entry = portfolio[asset]
            current_perf = (current - entry['avg_price']) / entry['avg_price']

            if current_perf <= -self.sl_max or current_perf >= self.tp_max:
                action.update({'type': 'sell', 'quantity': entry['quantity']})
                return action

        return action
