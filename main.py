# region imports
from AlgorithmImports import *

from universe import BigTechUniverseSelectionModel
from alpha import GaussianNaiveBayesAlphaModel
from portfolio import ClampedEqualWeightingPortfolioConstructionModel
# endregion

class GaussianNaiveBayesClassificationAlgorithm(QCAlgorithm):

    undesired_symbols_from_previous_deployment = []
    checked_symbols_from_previous_deployment = False

    def initialize(self):
        self.set_end_date(datetime.now())
        self.set_start_date(self.end_date - timedelta(10*365))
        self.set_cash(50000)

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.set_security_initializer(BrokerageModelSecurityInitializer(self.brokerage_model, FuncSecuritySeeder(self.get_last_known_prices)))

        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self.universe_settings.schedule.on(self.date_rules.week_start())
        self.set_universe_selection(BigTechUniverseSelectionModel(self.universe_settings, self.get_parameter("universe_size", 15)))
        
        self.set_alpha(GaussianNaiveBayesAlphaModel())
        
        self.week = -1
        # Equal-weight positions with clamps: >=3% of portfolio and <=25% of investable cash
        self.set_portfolio_construction(ClampedEqualWeightingPortfolioConstructionModel(
            self.rebalance_func,
            min_weight=0.03,
            max_cash_fraction=0.25
        ))

        self.set_risk_management(NullRiskManagementModel())
        '''self.set_risk_management(TrailingStopRiskManagementModel(0.1)) '''
        
        '''self.set_execution(ImmediateExecutionModel()) '''
        self.set_execution(StandardDeviationExecutionModel(deviations=3,period=60,resolution=Resolution.MINUTE)) 
        '''self.set_execution(SpreadExecutionModel()) '''


        self.set_warm_up(timedelta(10))
        'Set the Benchmark to be SPY '
        self.set_benchmark("SPY")

    def rebalance_func(self, time):
        week = self.time.isocalendar()[1]
        if self.week != week and not self.is_warming_up and self.current_slice.quote_bars.count > 0:
            self.week = week
            return time
        return None
        
    def on_data(self, data):
        # Exit positions that aren't backed by existing insights.
        # If you don't want this behavior, delete this method definition.
        if not self.is_warming_up and not self.checked_symbols_from_previous_deployment:
            for security_holding in self.portfolio.values():
                if not security_holding.invested:
                    continue
                symbol = security_holding.symbol
                if not self.insights.has_active_insights(symbol, self.utc_time):
                    self.undesired_symbols_from_previous_deployment.append(symbol)
            self.checked_symbols_from_previous_deployment = True
        
        for symbol in self.undesired_symbols_from_previous_deployment:
            if self.is_market_open(symbol):
                self.liquidate(symbol, tag="Holding from previous deployment that's no longer desired")
                self.undesired_symbols_from_previous_deployment.remove(symbol)
