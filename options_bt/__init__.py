"""Options Backtesting Engine Package"""
from .pricing import Black76, ImpliedVolatilitySolver, VolatilitySurfaceMetrics, GreeksResult
from .chain_parser import OptionsChainParser
from .strike_matrix import (
    StrikeMatrixEngine,
    StrikeProfile,
    StraddleStrikes,
    StrangleStrikes,
    IronCondorStrikes,
)
from .leg_cutter import LegCutter, LegCutConfig, LegState
from .oi_fader import OIFaderEngine, OIFaderSignal
from .oi_analyzer import (
    OpenInterestAnalyzer,
    BuildupType,
    BuildupRecord,
    PCRMetrics,
    MaxPainResult,
    StrikeOISummary,
)
from .pnl_attribution import GreekPnLAttributor, BarAttribution
from .backtester import OptionsBacktester, SessionResult, BacktestMetrics
from .strangle_engine import StrangleEngine, StrangleConfig, StrangleSessionResult
from .iron_condor import IronCondorEngine, IronCondorConfig, IronCondorSessionResult
from .surface import (
    VolatilitySurfaceEngine,
    SmileSlice,
    SmilePoint,
    VolatilitySurfaceMesh,
)
from .visualizer import OptionsVisualizer
from .spike_fader import (
    SpikeFaderEngine,
    SpikeFaderConfig,
    SpikeTradeRecord,
    SpikeFaderSummary,
)

__all__ = [
    "Black76",
    "ImpliedVolatilitySolver",
    "VolatilitySurfaceMetrics",
    "GreeksResult",
    "OptionsChainParser",
    "StrikeMatrixEngine",
    "StrikeProfile",
    "StraddleStrikes",
    "StrangleStrikes",
    "IronCondorStrikes",
    "LegCutter",
    "LegCutConfig",
    "LegState",
    "OIFaderEngine",
    "OIFaderSignal",
    "OpenInterestAnalyzer",
    "BuildupType",
    "BuildupRecord",
    "PCRMetrics",
    "MaxPainResult",
    "StrikeOISummary",
    "GreekPnLAttributor",
    "BarAttribution",
    "OptionsBacktester",
    "SessionResult",
    "BacktestMetrics",
    "StrangleEngine",
    "StrangleConfig",
    "StrangleSessionResult",
    "IronCondorEngine",
    "IronCondorConfig",
    "IronCondorSessionResult",
    "VolatilitySurfaceEngine",
    "SmileSlice",
    "SmilePoint",
    "VolatilitySurfaceMesh",
    "OptionsVisualizer",
    "SpikeFaderEngine",
    "SpikeFaderConfig",
    "SpikeTradeRecord",
    "SpikeFaderSummary",
]
