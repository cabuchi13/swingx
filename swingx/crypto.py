"""Universo y contexto para cripto.

Por que cripto y no acciones
----------------------------
El obstaculo estructural que rompio todo en acciones fue el gap: 177 salidas
por gap a -1.00R promedio, y la imposibilidad de cerrar fuera de horario en
BingX. Cripto opera 24/7: el stop se ejecuta siempre. Eso no hace la estrategia
rentable, pero elimina el riesgo que el sizing no podia controlar.

Ademas es un dataset que NUNCA tocamos. Los 8 años de acciones estadounidenses
ya los gastamos en dos familias de hipotesis, cinco variantes, 243 configuraciones
y diez tests de filtros. Cada mirada adicional ahi vale menos. Aca empezamos
limpio.

Diferencias que importan para el backtest
----------------------------------------
  - 365 barras por año en vez de 252: mas datos por unidad de tiempo.
  - Sin earnings: desaparece el bloqueo por reportes.
  - Referencia de regimen: BTC, no SPY. El mercado cripto se mueve con bitcoin.
  - Volatilidad mucho mas alta: el tope de gap del sizing se vuelve el limitante
    casi siempre, y eso esta bien.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind

# Perpetuos con volumen real en BingX y con historico suficiente en Yahoo.
# Los memecoins muy nuevos quedan afuera: sin historia no hay backtest.
UNIVERSO_CRYPTO = [
    # Mayores
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD",
    "BCH-USD", "XLM-USD", "ATOM-USD", "ETC-USD", "HBAR-USD", "ICP-USD",
    "FIL-USD", "VET-USD", "ALGO-USD",
    # DeFi e infraestructura
    "UNI-USD", "AAVE-USD", "MKR-USD", "GRT-USD", "INJ-USD", "RUNE-USD",
    "SNX-USD", "CRV-USD", "COMP-USD", "LDO-USD",
    # Capa 2 y nuevas cadenas
    "ARB-USD", "OP-USD", "NEAR-USD", "APT-USD", "SUI-USD", "SEI-USD",
    "TIA-USD", "IMX-USD", "STX-USD",
    # Alta beta y memes con historia
    "SHIB-USD", "PEPE-USD", "FET-USD", "RENDER-USD", "THETA-USD",
    "SAND-USD", "MANA-USD", "AXS-USD", "CHZ-USD", "GALA-USD",
    "EOS-USD", "XTZ-USD", "NEO-USD", "IOTA-USD", "ZEC-USD", "DASH-USD",
]

REFERENCIA_REGIMEN = "BTC-USD"   # bitcoin manda el regimen del mercado cripto


def regimen_cripto(btc: pd.DataFrame, sma_len: int = 200, pendiente: int = 20) -> pd.DataFrame:
    """Estado del mercado cripto segun bitcoin.

    Mismo criterio que usamos con SPY: BTC sobre su SMA200 y esa media subiendo.
    """
    c = btc["Close"]
    sma = ind.sma(c, sma_len)
    out = pd.DataFrame(index=btc.index)
    out["regime_ok"] = ((c > sma) & (sma > sma.shift(pendiente))).fillna(False)
    out["mkt_ret_5"] = c.pct_change(5)
    out["mkt_ret_63"] = c.pct_change(63)
    return out


def es_cripto(ticker: str) -> bool:
    return ticker.upper().endswith("-USD") and not ticker.upper().startswith(("BRK",))
