"""Configuracion del sistema: universo, parametros del setup y costos de BingX."""
from __future__ import annotations
from dataclasses import dataclass

# --- Universo -----------------------------------------------------------
# Acciones con contrato en BingX (perpetuos sobre acciones / standard futures).
# Verifica la lista en la plataforma antes de operar: BingX agrega y delista pares.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "TSM", "ARM", "MU", "AVGO",
    "PYPL", "SOFI", "HOOD", "DKNG", "UBER", "ABNB", "SHOP", "COIN",
    "V", "MA", "JNJ", "PFE", "WMT", "KO", "BRK-B", "UNH", "BAC", "XOM",
    "PLTR", "NIO", "F", "CCL", "MSTR",
]

# Subconjunto defensivo: menor riesgo de gap, apto para apalancamiento mayor.
LOW_VOL_TIER = {"V", "MA", "JNJ", "PFE", "WMT", "KO", "BRK-B", "UNH", "BAC", "XOM", "AAPL", "MSFT"}

SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "consumer", "META": "tech",
    "NVDA": "semis", "TSLA": "consumer", "NFLX": "tech", "AMD": "semis", "INTC": "semis",
    "TSM": "semis", "ARM": "semis", "MU": "semis", "AVGO": "semis", "PYPL": "fintech",
    "SOFI": "fintech", "HOOD": "fintech", "DKNG": "consumer", "UBER": "consumer",
    "ABNB": "consumer", "SHOP": "tech", "COIN": "fintech", "V": "fintech", "MA": "fintech",
    "JNJ": "health", "PFE": "health", "WMT": "consumer", "KO": "consumer",
    "BRK-B": "financials", "UNH": "health", "BAC": "financials", "XOM": "energy",
    "PLTR": "tech", "NIO": "consumer", "F": "consumer", "CCL": "consumer", "MSTR": "fintech",
}


@dataclass
class SetupParams:
    """Parametros del setup 'pullback en tendencia'."""
    sma_fast: int = 50
    sma_slow: int = 200
    rsi_len: int = 2
    rsi_max: float = 15.0
    bb_len: int = 20
    bb_k: float = 2.0
    pullback_min: float = 0.03
    pullback_max: float = 0.18
    lookback_high: int = 20
    max_pct_below_fast: float = 0.03
    min_dollar_volume: float = 50e6
    atr_len: int = 14
    min_atr_pct: float = 0.012
    max_atr_pct: float = 0.060
    min_adx: float = 0.0
    earnings_blackout_days: int = 10
    atr_stop_mult: float = 1.5
    swing_low_lookback: int = 10
    target_r_1: float = 1.5
    partial_at_t1: float = 0.5
    trail_atr_mult: float = 2.5
    time_stop_days: int = 12


@dataclass
class BingXCosts:
    """Costos de BingX sobre contratos de acciones.

    OJO: funding y fees se cobran sobre el NOCIONAL, no sobre tu margen.
    Con 10x, un costo de 0.1% del nocional es 1% de tu margen.
    """
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    funding_rate: float = 0.0001
    funding_intervals_per_day: int = 6
    max_leverage: float = 25.0


@dataclass
class RiskParams:
    """Reglas de riesgo. Esto es el sistema: el setup solo elige que mirar."""
    risk_per_trade: float = 0.01
    max_gap_pct: float = 0.20
    max_gap_loss_pct: float = 0.05
    liq_buffer: float = 1.6
    maintenance_margin_rate: float = 0.005
    max_margin_pct: float = 0.30
    max_concurrent: int = 4
    max_per_sector: int = 2
