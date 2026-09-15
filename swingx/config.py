"""Configuracion del sistema: universo, parametros del setup y costos de BingX."""
from __future__ import annotations
from dataclasses import dataclass

# --- Universo -----------------------------------------------------------
# Extraido de la plataforma en vivo (bingx.com/en/market/stocks, filtro
# "US Stocks", 4 paginas, 117 instrumentos) el 15/09/2026.
#
# Excluidos a proposito:
#   - ETFs apalancados 3x (SOXL, SOXS, TQQQ, SQQQ): ya vienen apalancados;
#     sumarles el nuestro multiplica el riesgo dos veces.
#   - Sinteticos sin mercado publico (SPCX/SpaceX, PURRUS, BMNR): no hay
#     historico real contra el cual validar nada.
#   - Indices amplios (SPY, QQQ): son la referencia de regimen, no candidatos.
#   - No estadounidenses (HYUNDAI, SAMSUNG, SKHYNIX): otro huso horario.
#
# Los nombres con poco historico (IPOs recientes) los descarta sola la capa
# de datos, que exige mas de 260 ruedas.
UNIVERSE = [
    # Megacaps y software
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "ORCL", "CRM", "NOW",
    "IBM", "CSCO", "PANW", "APP", "SNOW", "PATH", "RDDT",
    # Semis y hardware
    "NVDA", "AMD", "INTC", "TSM", "ASML", "AVGO", "MU", "MRVL", "QCOM",
    "AMAT", "LRCX", "TXN", "ARM", "SMCI", "DELL", "HPQ", "WDC", "STX",
    "SNDK", "CRDO", "COHR", "LITE", "GLW", "AAOI", "KOPN", "NBIS", "CBRS",
    # Fintech, cripto y bancos
    "COIN", "HOOD", "SOFI", "MSTR", "MARA", "CRCL", "JPM", "GS", "MS", "BRK-B",
    # Infraestructura de IA
    "CRWV", "IREN", "APLD", "VRT",
    # Salud y consumo
    "JNJ", "LLY", "NVO", "COST", "MAR", "RACE", "AMC", "GME",
    # Energia y materiales
    "XOM", "OXY", "COP", "SLB", "LNG", "MP",
    # Industria, espacio y defensa
    "GE", "LMT", "RKLB", "ASTS", "RDW", "SIDU", "SPCE", "FLY",
    # Autos y movilidad
    "TSLA", "RIVN", "F", "OUST",
    # Energia limpia y nuclear
    "PLUG", "FLNC", "SMR",
    # Cuantica e IA especulativa
    "IONQ", "QBTS", "RGTI", "QUBT", "ARQQ", "BBAI", "INFQ", "LWLG",
    # Telecom y otros
    "NOK", "BB", "ONDS", "USAR", "BE",
    # ETFs sectoriales sin apalancar — no tienen earnings
    "XLE", "XOP", "IGV", "EWT",
]

# Traduccion ticker real -> simbolo en BingX, donde difieren.
BINGX_SYMBOL = {
    "AMD": "AMDUS", "SOFI": "SOFIUS", "OXY": "OXYUS", "COP": "COPUS",
    "MP": "MPUS", "STX": "STXUS", "NOK": "NOKUS", "SNOW": "SNOWUS",
    "LMT": "LMTUS", "AMC": "AMCUS", "F": "FUS", "BB": "BBUS",
    "BRK-B": "BRKB", "FLY": "FLYUS",
}

# Subconjunto defensivo: menor riesgo de gap, tolera apalancamiento mayor.
LOW_VOL_TIER = {"AAPL", "MSFT", "JNJ", "XOM", "BRK-B", "JPM", "CSCO",
                "IBM", "TXN", "COST", "XLE"}

_SEC = {
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NFLX", "ORCL", "CRM", "NOW", "IBM",
             "CSCO", "PANW", "APP", "SNOW", "PATH", "RDDT", "IGV"],
    "semis": ["NVDA", "AMD", "INTC", "TSM", "ASML", "AVGO", "MU", "MRVL", "QCOM",
              "AMAT", "LRCX", "TXN", "ARM", "SMCI", "DELL", "HPQ", "WDC", "STX",
              "SNDK", "CRDO", "COHR", "LITE", "GLW", "AAOI", "KOPN", "CBRS", "EWT"],
    "fintech": ["COIN", "HOOD", "SOFI", "MSTR", "MARA", "CRCL", "JPM", "GS", "MS", "BRK-B"],
    "infra-ia": ["CRWV", "IREN", "APLD", "NBIS", "VRT"],
    "health": ["JNJ", "LLY", "NVO"],
    "consumer": ["AMZN", "COST", "MAR", "RACE", "AMC", "GME"],
    "energia": ["XOM", "OXY", "COP", "SLB", "LNG", "MP", "XLE", "XOP"],
    "espacio": ["GE", "LMT", "RKLB", "ASTS", "RDW", "SIDU", "SPCE", "FLY"],
    "autos": ["TSLA", "RIVN", "F", "OUST"],
    "energia-limpia": ["PLUG", "FLNC", "SMR"],
    "cuantica": ["IONQ", "QBTS", "RGTI", "QUBT", "ARQQ", "BBAI", "INFQ", "LWLG"],
    "telecom": ["NOK", "BB", "ONDS", "USAR", "BE"],
}
SECTOR = {t: s for s, ts in _SEC.items() for t in ts}


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
    min_notional: float = 2.0            # verificado: Min. Trade Value = 2 USDT
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    funding_rate: float = 0.0001
    funding_intervals_per_day: int = 3   # verificado en la plataforma: intervalo de 8H
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
