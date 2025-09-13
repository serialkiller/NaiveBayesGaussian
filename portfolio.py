from AlgorithmImports import *


class ClampedEqualWeightingPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):
    """
    Confidence-weighted sizing for active long insights with clamps:
    - Base weight per symbol is derived from the insight's `weight` if present, otherwise `confidence`.
      If neither is provided, defaults to 1 for equal weighting.
    - Clamp per-position to a minimum of `min_weight` (default 3% of portfolio value).
    - Clamp per-position to a maximum based on investable cash: `max_cash_fraction * (cash / total_portfolio_value)`.

    Notes:
    - If clamps conflict (e.g., not enough cash), the cash cap wins.
    - If the sum of targets exceeds 100% after clamping, targets are scaled down proportionally.
    """

    def __init__(self, rebalance=None, min_weight: float = 0.03, max_cash_fraction: float = 0.25):
        super().__init__(rebalance)
        self.min_weight = float(min_weight)
        self.max_cash_fraction = float(max_cash_fraction)

    def determine_target_percent(self, active_insights):
        # Keep only the latest long insight per symbol
        latest_by_symbol = {}
        for ins in active_insights:
            if ins.direction == InsightDirection.UP:
                latest_by_symbol[ins.symbol] = ins

        if not latest_by_symbol:
            return {}

        # Derive raw scores from insight weight or confidence
        raw_scores = {}
        for sym, ins in latest_by_symbol.items():
            score = getattr(ins, 'weight', None)
            if score is None or score <= 0:
                score = getattr(ins, 'confidence', None)
            if score is None or score <= 0:
                score = 1.0  # fallback to equal weighting
            raw_scores[ins] = float(score)

        total_score = sum(raw_scores.values())
        if total_score <= 0:
            # All scores zero/invalid – fall back to equal weights
            n = len(latest_by_symbol)
            for ins in list(raw_scores.keys()):
                raw_scores[ins] = 1.0 / n
            total_score = 1.0

        # Normalize scores to weights summing to 1
        norm_weights = {ins: score / total_score for ins, score in raw_scores.items()}

        # Compute dynamic cash-based max cap in portfolio weight terms
        total_value = float(self.algorithm.portfolio.total_portfolio_value)
        cash = float(self.algorithm.portfolio.cash)
        max_weight_cash_capped = 0.0
        if total_value > 0.0:
            max_weight_cash_capped = max(0.0, self.max_cash_fraction * (cash / total_value))

        # Apply per-position clamps
        clamped = {}
        for ins, w in norm_weights.items():
            w = max(self.min_weight, w)
            w = min(w, max_weight_cash_capped)
            clamped[ins] = w

        # If sum exceeds 100%, scale down proportionally
        total = sum(clamped.values())
        if total > 1.0 and total > 0:
            scale = 1.0 / total
            for ins in list(clamped.keys()):
                clamped[ins] *= scale

        return clamped
