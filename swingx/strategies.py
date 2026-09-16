"""Catalogo de familias de estrategias, todas declaradas ANTES de testear.

Cada familia es una hipotesis distinta sobre por que el precio se mueve, no una
variacion de parametros de la misma idea. Las tres primeras ya las probamos y
fallaron; quedan como linea de base para comparar.

Cada estrategia devuelve una mascara booleana de entradas. La mecanica de salida,
los costos y el control contra azar son IDENTICOS para todas, asi que la unica
diferencia medida es la senal.

REGLA DEL TORNEO: probar N familias multiplica por N la chance de encontrar un
falso positivo. Por eso el umbral se corrige por Bonferroni en tournament.py:
cuantas mas familias probemos, mas alto tiene que saltar cada una.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SetupParams
from .signals import compute_features
from . import indicators as ind


# ---------------------------------------------------------------------------
# Cada funcion recibe (df, ctx, f) y devuelve una Series booleana.
#   df  = OHLCV crudo
#   ctx = contexto transversal (rs_rank, regime_ok, rel_drop_5) o None
#   f   = features ya calculados (SMA, RSI, ATR, Bollinger, ADX...)
# ---------------------------------------------------------------------------

def mean_reversion(df, ctx, f):
    """FAMILIA 1 — Comprar sobreventa de corto en tendencia alcista.
    Hipotesis: en una tendencia sana, las caidas rapidas se compran.
    Estado: PROBADA Y FALLIDA. Queda como linea de base."""
    c = f["Close"]
    return ((c > f["sma_slow"]) & (f["sma_fast"] > f["sma_slow"])
            & (f["rsi_fast"] < 15) & (f["pct_vs_fast"] > -0.03))


def momentum_breakout(df, ctx, f):
    """FAMILIA 2 — Ruptura de maximo de 20 dias en acciones fuertes.
    Hipotesis: la fuerza relativa persiste.
    Estado: PROBADA Y FALLIDA. Linea de base."""
    c = f["Close"]
    hi = c.rolling(20, min_periods=20).max()
    m = (c >= hi * 0.999) & (c > f["sma_slow"]) & (f["sma_fast"] > f["sma_slow"])
    if ctx is not None and "rs_rank" in ctx:
        m = m & (ctx["rs_rank"].reindex(f.index) >= 0.70).fillna(False)
    return m


def donchian_trend(df, ctx, f):
    """FAMILIA 3 — Donchian clasico: ruptura de 55 ruedas.
    Hipotesis (tortugas): las tendencias largas duran mas de lo que parece
    razonable. Es momentum pero con ventana mucho mas larga, lo que lo hace
    una apuesta distinta: menos senales, mas duracion, menos sensible al ruido.
    NUEVA — nunca testeada."""
    c = f["Close"]
    hi55 = c.rolling(55, min_periods=55).max()
    return (c >= hi55 * 0.999) & (c > f["sma_slow"])


def squeeze_expansion(df, ctx, f):
    """FAMILIA 4 — Compresion de volatilidad seguida de expansion.
    Hipotesis: la volatilidad es ciclica. Periodos de rango estrecho preceden
    movimientos grandes; la direccion la define la ruptura. Es una apuesta a la
    ESTRUCTURA de la volatilidad, no al precio — ortogonal a las anteriores.
    NUEVA — nunca testeada."""
    c = f["Close"]
    ancho = (f["bb_up"] - f["bb_low"]) / f["bb_mid"]
    # el ancho de bandas esta en el quintil mas bajo de los ultimos 6 meses
    comprimido = ancho <= ancho.rolling(126, min_periods=126).quantile(0.20)
    # y hoy rompe hacia arriba saliendo de la compresion
    rompe = c > c.shift(1).rolling(10, min_periods=10).max()
    return (comprimido.shift(1).fillna(False) & rompe & (c > f["sma_slow"])).fillna(False)


def volatility_breakout(df, ctx, f):
    """FAMILIA 5 — Cierre por encima de la apertura mas k*ATR.
    Hipotesis: un dia de rango excepcionalmente grande y cierre fuerte indica
    entrada de capital institucional que continua al dia siguiente. Es una
    apuesta intradiaria a la FUERZA del dia, no a la tendencia.
    NUEVA — nunca testeada."""
    o, c = f["Open"], f["Close"]
    empuje = (c - o) > 1.0 * f["atr"]
    cierre_fuerte = (c - f["Low"]) / (f["High"] - f["Low"]).replace(0, np.nan) > 0.75
    return (empuje & cierre_fuerte & (c > f["sma_slow"])).fillna(False)


def ma_pullback_strong(df, ctx, f):
    """FAMILIA 6 — Retroceso a la EMA20 en tendencia fuerte (ADX alto).
    Hipotesis: distinta de la familia 1 en dos cosas que importan: exige
    tendencia FUERTE medida por ADX (no solo alcista), y compra un retroceso
    poco profundo a una media rapida, no sobreventa extrema.
    NUEVA — nunca testeada."""
    c = f["Close"]
    ema20 = ind.ema(c, 20)
    toca = (f["Low"] <= ema20 * 1.01) & (c > ema20 * 0.99)
    return (toca & (f["adx"] > 25) & (c > f["sma_slow"])
            & (f["sma_fast"] > f["sma_slow"])).fillna(False)


def new_high_pullback(df, ctx, f):
    """FAMILIA 7 — Maximo de 52 semanas reciente, luego pausa breve.
    Hipotesis: el 'efecto maximo de 52 semanas' esta documentado: las acciones
    cerca de su maximo anual tienden a seguir subiendo porque los inversores
    anclan al maximo previo y subreaccionan a buenas noticias.
    NUEVA — nunca testeada."""
    c = f["Close"]
    hi252 = c.rolling(252, min_periods=252).max()
    cerca_del_maximo = c >= hi252 * 0.95
    hizo_maximo_reciente = (c >= hi252 * 0.999).rolling(20, min_periods=1).max() > 0
    pausa = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    return (cerca_del_maximo & hizo_maximo_reciente & pausa & (c > f["sma_slow"])).fillna(False)


# Registro. El orden no importa; el torneo las corre todas.
FAMILIAS = {
    "1_mean_reversion":    (mean_reversion,     "Sobreventa en tendencia (base, ya fallida)"),
    "2_momentum_breakout": (momentum_breakout,  "Ruptura 20d en acciones fuertes (base, ya fallida)"),
    "3_donchian_55":       (donchian_trend,     "Ruptura de 55 ruedas, estilo tortugas"),
    "4_squeeze":           (squeeze_expansion,  "Compresion de volatilidad y expansion"),
    "5_vol_breakout":      (volatility_breakout,"Dia de rango grande con cierre fuerte"),
    "6_ema20_adx":         (ma_pullback_strong, "Retroceso a EMA20 con ADX alto"),
    "7_max_52s":           (new_high_pullback,  "Pausa cerca del maximo de 52 semanas"),
}


def mascara(nombre: str, df: pd.DataFrame, ctx, p: SetupParams | None = None) -> pd.Series:
    """Calcula la mascara de una familia, con features compartidos."""
    p = p or SetupParams()
    fn = FAMILIAS[nombre][0]
    f = compute_features(df, p)
    m = fn(df, ctx, f)
    # filtros minimos comunes a todas, para que compitan en igualdad de condiciones
    liquidez = f["dvol"] > p.min_dollar_volume
    vol_util = f["atr_pct"].between(0.008, 0.10)
    return (m & liquidez & vol_util).fillna(False).reindex(f.index).fillna(False)
