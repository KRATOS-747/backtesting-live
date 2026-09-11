"""Equity Backtesting Engine Package"""
from .pit_universe import PointInTimeUniverse
from .pca_factor_model import PCAResidualMomentumEngine, PCAStrategyConfig
from .factor_optimizer import FactorOptimizer, FactorPortfolioWeights
from .costs_tax import EquityCostModel, EquityTradeCost
from .pca_strategy import (
    PCABacktestV1_BaseMomentum,
    PCABacktestV2_ResidualExtraction,
    PCABacktestV3_ATRBufferedPCA,
)
from .alpha_strategy import (
    AlphaMomentumV1_CompositeZScore,
    AlphaMomentumV2_SortinoRankBuffer,
    AlphaMomentumV3_FullWithATR,
    AlphaStrategyConfig,
    AlphaBacktestResult,
)
from .price_action_momentum_strategy import (
    PriceActionConfig,
    PriceActionTrade,
    PriceActionSummary,
    PriceActionMomentumStrategy,
)
from .indicator_momentum_strategy import (
    IndicatorConfig,
    IndicatorTrade,
    IndicatorSummary,
    IndicatorMomentumStrategy,
)
from .ml_momentum_strategy import (
    MLMomentumConfig,
    MLTrade,
    MLSummary,
    MLMomentumStrategy,
)

__all__ = [
    "PointInTimeUniverse",
    "PCAResidualMomentumEngine",
    "PCAStrategyConfig",
    "FactorOptimizer",
    "FactorPortfolioWeights",
    "EquityCostModel",
    "EquityTradeCost",
    "PCABacktestV1_BaseMomentum",
    "PCABacktestV2_ResidualExtraction",
    "PCABacktestV3_ATRBufferedPCA",
    "AlphaMomentumV1_CompositeZScore",
    "AlphaMomentumV2_SortinoRankBuffer",
    "AlphaMomentumV3_FullWithATR",
    "AlphaStrategyConfig",
    "AlphaBacktestResult",
    "PriceActionConfig",
    "PriceActionTrade",
    "PriceActionSummary",
    "PriceActionMomentumStrategy",
    "IndicatorConfig",
    "IndicatorTrade",
    "IndicatorSummary",
    "IndicatorMomentumStrategy",
    "MLMomentumConfig",
    "MLTrade",
    "MLSummary",
    "MLMomentumStrategy",
]

