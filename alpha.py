from AlgorithmImports import *

from symbol_data import SymbolData

from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

class GaussianNaiveBayesAlphaModel(AlphaModel):
    def __init__(self):
        self._symbol_data_by_symbol = {}
        self._week = -1

    def update(self, algorithm, data):
        week = data.time.isocalendar()[1]
        if self._week == week or data.quote_bars.count == 0:
            return []
        
        # Retrain weekly to incorporate latest features/labels
        self.train()

        long_symbols = []
        tradable_symbols = [s for s, sd in self._symbol_data_by_symbol.items() if sd.is_ready]

        for symbol in tradable_symbols:
            symbol_data = self._symbol_data_by_symbol[symbol]
            # Predict with only that symbol's most recent features, as a 2D DataFrame
            features = symbol_data.features_by_day.iloc[[-1]]
            if symbol_data.model is not None and len(features.columns) == symbol_data.model.n_features_in_:
                # Use class probabilities for confidence-weighted sizing
                try:
                    proba = symbol_data.model.predict_proba(features)[0]
                    classes = list(symbol_data.model.classes_)
                    if 1 in classes:
                        p_up = float(proba[classes.index(1)])
                    else:
                        p_up = 0.0
                except Exception:
                    p_up = 0.0

                # Go long only if probability of UP > 55%
                if p_up > 0.55:
                    long_symbols.append((symbol, p_up))

        if not long_symbols:
            return []
        self._week = week
        # Emit probability-weighted insights; PCM will normalize/clamp
        insights = []
        for symbol, p_up in long_symbols:
            insights.append(Insight.price(symbol, Expiry.ONE_MONTH, InsightDirection.UP, weight=p_up, confidence=p_up))
        return insights

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
                    # Fit GaussianNB with probability calibration when feasible; otherwise fallback
                    try:
                        counts = y.value_counts()
                        if len(counts) >= 2:
                            # Prefer 3-fold, but degrade to 2-fold if class counts are small
                            fitted = False
                            for cv in (3, 2):
                                if counts.min() >= cv:
                                    try:
                                        model = CalibratedClassifierCV(GaussianNB(), method='sigmoid', cv=cv)
                                        symbol_data.model = model.fit(X, y)
                                        fitted = True
                                        break
                                    except Exception:
                                        continue
                            if not fitted:
                                symbol_data.model = GaussianNB().fit(X, y)
                        else:
                            symbol_data.model = GaussianNB().fit(X, y)
                    except Exception:
                        # Fallback to plain GaussianNB in case calibration is unavailable
                        symbol_data.model = GaussianNB().fit(X, y)