"""Contexto de mercado: las dos piezas que le faltan al setup.

El setup actual mira cada accion aislada. Eso ignora dos cosas que la
literatura documenta como fuentes de edge reales, no ajustes de parametros:

1. REGIMEN DE MERCADO
   Comprar caidas funciona en mercados alcistas y falla catastroficamente en
   bajistas. Es el mismo setup con resultado opuesto segun el contexto. Filtrar
   por el estado del indice no mejora las ganadoras: elimina el periodo donde
   la estrategia pierde en serie. Ataca directamente el problema de la deriva:
   si solo operamos cuando el mercado sube, dejamos de depender de que suba.

2. FUERZA RELATIVA (momentum transversal)
   Es de las anomalias mejor documentadas y mas persistentes: las acciones que
   vienen rindiendo mejor que sus pares tienden a seguir haciendolo. Nuestro
   setup usa umbrales absolutos — cualquier accion que pase el filtro entra.
   Rankear contra el universo es distinto: compramos el pullback de la MAS
   fuerte, no de cualquiera que este sobrevendida.

3. CAIDA RELATIVA AL MERCADO
   Una accion que cae 6% mientras el mercado cae 5% no esta sobrevendida: esta
   siguiendo al mercado. Una que cae 6% mientras el mercado sube 1% tiene un
   problema propio, o una oportunidad propia. No es lo mismo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def market_regime(spy: pd.DataFrame, sma_len: int = 200, slope_len: int = 20) -> pd.DataFrame:
    """Estado del mercado a partir del indice de referencia (SPY).

    Devuelve un DataFrame con:
      regime_ok  : el indice esta sobre su SMA200 y esa media no viene cayendo
      mkt_ret_5  : retorno de 5 dias del indice (para la caida relativa)
    """
    c = spy["Close"]
    sma = ind.sma(c, sma_len)
    subiendo = sma > sma.shift(slope_len)
    out = pd.DataFrame(index=spy.index)
    out["regime_ok"] = ((c > sma) & subiendo).fillna(False)
    out["mkt_ret_5"] = c.pct_change(5)
    out["mkt_ret_63"] = c.pct_change(63)
    return out


def relative_strength(data: dict, lookback: int = 63) -> pd.DataFrame:
    """Ranking transversal: percentil del retorno de cada accion contra el universo.

    1.0 = la mas fuerte del universo ese dia. 0.0 = la mas debil.
    """
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()}).sort_index()
    ret = closes.pct_change(lookback)
    # rank por fila (por fecha), normalizado a 0-1
    return ret.rank(axis=1, pct=True)


def build_context(data: dict, spy: pd.DataFrame | None, rs_lookback: int = 63) -> dict:
    """Arma, por ticker, las series de contexto alineadas a su indice."""
    rs = relative_strength(data, rs_lookback)
    reg = market_regime(spy) if spy is not None and len(spy) else None

    ctx = {}
    for t, df in data.items():
        c = pd.DataFrame(index=df.index)
        c["rs_rank"] = rs[t].reindex(df.index) if t in rs.columns else np.nan
        if reg is not None:
            r = reg.reindex(df.index).ffill()
            c["regime_ok"] = r["regime_ok"].fillna(False)
            c["mkt_ret_5"] = r["mkt_ret_5"]
        else:
            c["regime_ok"] = True
            c["mkt_ret_5"] = 0.0
        # caida de la accion en 5 dias menos la del mercado:
        # negativo = cayo mas que el mercado (debilidad propia)
        c["rel_drop_5"] = df["Close"].pct_change(5) - c["mkt_ret_5"]
        ctx[t] = c
    return ctx


def apply_context(armed: pd.Series, ctx: pd.DataFrame, *,
                  use_regime: bool = False,
                  min_rs_rank: float = 0.0,
                  max_rel_drop: float | None = None) -> pd.Series:
    """Filtra una mascara de señales con las condiciones de contexto activadas."""
    m = armed.copy()
    if ctx is None or not len(ctx):
        return m
    c = ctx.reindex(m.index)
    if use_regime:
        m = m & c["regime_ok"].fillna(False).astype(bool)
    if min_rs_rank > 0:
        m = m & (c["rs_rank"] >= min_rs_rank).fillna(False)
    if max_rel_drop is not None:
        # exigir que NO haya caido mucho mas que el mercado:
        # rel_drop_5 por encima del umbral negativo
        m = m & (c["rel_drop_5"] >= max_rel_drop).fillna(False)
    return m.fillna(False)


# Las variantes que vamos a testear. Cada una tiene una hipotesis declarada
# ANTES de ver el resultado. Son cuatro comparaciones, no una busqueda.
VARIANTES = {
    "base": {
        "hipotesis": "El setup actual, tal como esta.",
        "cfg": {},
    },
    "regimen": {
        "hipotesis": "Filtrar por SPY sobre su SMA200 elimina el periodo donde "
                     "comprar caidas pierde en serie, y corta la dependencia de la deriva.",
        "cfg": {"use_regime": True},
    },
    "fuerza_relativa": {
        "hipotesis": "Comprar el pullback solo de acciones en el tercio mas fuerte "
                     "del universo aprovecha el momentum transversal.",
        "cfg": {"min_rs_rank": 0.67},
    },
    "regimen+fuerza": {
        "hipotesis": "Las dos capas juntas: operar solo en mercado alcista y solo "
                     "las acciones mas fuertes.",
        "cfg": {"use_regime": True, "min_rs_rank": 0.67},
    },
    "regimen+fuerza+relativa": {
        "hipotesis": "Ademas, descartar las que cayeron mucho mas que el mercado "
                     "(debilidad propia, no arrastre).",
        "cfg": {"use_regime": True, "min_rs_rank": 0.67, "max_rel_drop": -0.05},
    },
}
