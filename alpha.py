from AlgorithmImports import *

from symbol_data import SymbolData

from sklearn.naive_bayes import GaussianNB
from dateutil.relativedelta import relativedelta

class GaussianNaiveBayesAlphaModel(AlphaModel):
    def __init__(self):
        self._symbol_data_by_symbol = {}
        self._week = -1

    def update(self, algorithm, data):
        week = data.time.isocalendar()[1]
        if self._week == week or data.quote_bars.count == 0:
            return []

        long_symbols = []
        tradable_symbols = [s for s, sd in self._symbol_data_by_symbol.items() if sd.is_ready]

        for symbol in tradable_symbols:
            symbol_data = self._symbol_data_by_symbol[symbol]
            # Predict with only that symbol's most recent features, as a 2D DataFrame
            features = symbol_data.features_by_day.iloc[[-1]]
            if symbol_data.model is not None and len(features.columns) == symbol_data.model.n_features_in_:
                if symbol_data.model.predict(features)[0] == 1:
                    long_symbols.append(symbol)

        if not long_symbols:
            return []
        self._week = week
        weight = 1 / len(long_symbols)
        return [Insight.price(symbol, Expiry.ONE_MONTH, InsightDirection.UP, weight=weight) for symbol in long_symbols]

    def on_securities_changed(self, algorithm, changes):
        for security in changes.added_securities:
            self._symbol_data_by_symbol[security.symbol] = SymbolData(security, algorithm)
        for security in changes.removed_securities:
            symbol_data = self._symbol_data_by_symbol.pop(security.symbol, None)
            if symbol_data:
                symbol_data.dispose()
        self.train()

    def train(self):
        for symbol, symbol_data in self._symbol_data_by_symbol.items():
            if symbol_data.is_ready and not symbol_data.features_by_day.empty and not symbol_data.labels_by_day.empty:
                # Align features and labels
                idx = symbol_data.features_by_day.index.intersection(symbol_data.labels_by_day.index)
                X = symbol_data.features_by_day.loc[idx]
                y = symbol_data.labels_by_day.loc[idx]
                if not X.empty and not y.empty:
                    symbol_data.model = GaussianNB().fit(X, y)
