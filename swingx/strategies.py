"""Familias de estrategias, ahora derivadas de la evidencia publicada.

De donde sale cada una
----------------------
La version anterior de este archivo la escribi de memoria: RSI(2) de Connors,
tortugas, squeeze de Bollinger. Error. Estas vienen de literatura revisada,
con la cita al lado, y en particular de las CATEGORIAS que mejor replican.

Hou, Xue & Zhang replicaron 447 anomalias: 64% no pasan el 5% de significancia
y 85% no llegan a t>3. La unica categoria que aguanta razonablemente es
MOMENTUM, con 35% de fallas contra 93% de las de liquidez. Por eso casi todas
las familias nuevas son variantes de momentum: no por falta de imaginacion,
sino porque es lo unico que sobrevivio al escrutinio.

La construccion correcta
------------------------
Hurst, Ooi & Pedersen (AQR), "A Century of Evidence on Trend-Following":
momentum de SERIE TEMPORAL (el retorno pasado del propio activo predice el
suyo futuro) a 1, 3 y 12 meses, equiponderado, posiciones escaladas por
volatilidad. Sharpe ~0.40 sobre 67 mercados y 137 años.

Ojo con la diferencia: lo que yo habia construido era momentum TRANSVERSAL
(rankear contra el universo). Liu & Tsyvinski encontraron en cripto el de serie
temporal: en BTC, +1 desvio de suba hoy predice +0.33% mañana, y el efecto
persiste de 1 a 4 semanas.

Calibracion de expectativas
---------------------------
Sharpe 0.40 es lo que reporta la literatura de un siglo. Cualquier cosa por
encima de 1 en una estrategia publica es sobreajuste o ventana favorable hasta
que se demuestre lo contrario.

Y todo decae: McLean & Pontiff midieron 35% de caida post-publicacion. El
estudio de anomalias cripto bajo restricciones economicas encontro caidas de
9% a 76% entre periodos. Esto no se busca una vez y se opera para siempre.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import SetupParams
from .signals import compute_features
from . import indicators as ind


# Parametros de salida por tipo de estrategia.
# Las de tendencia necesitan meses, no dias: el error de la version anterior
# fue medir trend-following con stop por tiempo a 12 ruedas.
SALIDA_RAPIDA = dict(atr_stop_mult=1.5, target_r_1=1.5, trail_atr_mult=2.5,
                     time_stop_days=12, swing_low_lookback=10)
SALIDA_TENDENCIA = dict(atr_stop_mult=3.0, target_r_1=3.0, trail_atr_mult=4.0,
                        time_stop_days=120, swing_low_lookback=30)


# ---------------------------------------------------------------------------
# MOMENTUM DE SERIE TEMPORAL — la construccion de Hurst/AQR
# ---------------------------------------------------------------------------

def _tsmom(f: pd.DataFrame, dias: int) -> pd.Series:
    """Señal de momentum de serie temporal: retorno pasado positivo."""
    return f["Close"].pct_change(dias) > 0


def _cruce(señal: pd.Series) -> pd.Series:
    """Entrada en el momento en que la señal se enciende, no todos los dias."""
    return (señal & ~señal.shift(1).fillna(False)).fillna(False)


def tsmom_1m(df, ctx, f):
    """Momentum de serie temporal a 1 mes (21 ruedas).
    Hurst/AQR lo usan como el mas corto de los tres horizontes."""
    return _cruce(_tsmom(f, 21))


def tsmom_3m(df, ctx, f):
    """Momentum de serie temporal a 3 meses (63 ruedas).
    El horizonte intermedio de la construccion clasica."""
    return _cruce(_tsmom(f, 63))


def tsmom_12m(df, ctx, f):
    """Momentum de serie temporal a 12 meses (252 ruedas).
    El mas largo y el mas citado en la literatura de trend-following."""
    return _cruce(_tsmom(f, 252))


def tsmom_combo(df, ctx, f):
    """LA construccion de Hurst/AQR: combinacion equiponderada de 1, 3 y 12 meses.

    Entra cuando la mayoria de los horizontes coinciden en alcista y al menos
    uno acaba de encenderse. Es la version mas fiel a lo que reporta Sharpe 0.40
    sobre 137 años."""
    s1, s3, s12 = _tsmom(f, 21), _tsmom(f, 63), _tsmom(f, 252)
    mayoria = (s1.astype(int) + s3.astype(int) + s12.astype(int)) >= 2
    recien = _cruce(s1) | _cruce(s3) | _cruce(s12)
    return (mayoria & recien).fillna(False)


def tsmom_confirmado(df, ctx, f):
    """Momentum de serie temporal a 3 meses + confirmacion de fuerza reciente.

    Liu & Tsyvinski encontraron en cripto que el retorno reciente predice el
    proximo con persistencia de 1 a 4 semanas. Esto suma esa confirmacion de
    corto al horizonte intermedio."""
    base = _tsmom(f, 63)
    empuje = f["Close"].pct_change(7) > 0
    return (_cruce(base & empuje)).fillna(False)


# ---------------------------------------------------------------------------
# LINEAS DE BASE — ya probadas y fallidas, quedan para comparar
# ---------------------------------------------------------------------------

def base_mean_reversion(df, ctx, f):
    """Sobreversion en tendencia (Connors RSI-2). PROBADA Y FALLIDA.
    Queda para verificar que el torneo la sigue rechazando."""
    c = f["Close"]
    return ((c > f["sma_slow"]) & (f["sma_fast"] > f["sma_slow"])
            & (f["rsi_fast"] < 15) & (f["pct_vs_fast"] > -0.03))


def base_momentum_transversal(df, ctx, f):
    """Momentum TRANSVERSAL: ranking contra el universo. PROBADA Y FALLIDA.
    Es lo que yo habia construido por error en lugar del de serie temporal.
    Queda justamente para medir la diferencia entre las dos construcciones."""
    c = f["Close"]
    hi = c.rolling(20, min_periods=20).max()
    m = (c >= hi * 0.999) & (c > f["sma_slow"])
    if ctx is not None and "rs_rank" in ctx:
        m = m & (ctx["rs_rank"].reindex(f.index) >= 0.70).fillna(False)
    return m


# ---------------------------------------------------------------------------
# Registro: (funcion, hipotesis y fuente, parametros de salida)
# ---------------------------------------------------------------------------
FAMILIAS = {
    "tsmom_1m": (tsmom_1m,
                 "Momentum de serie temporal 1 mes (Hurst/AQR)", SALIDA_TENDENCIA),
    "tsmom_3m": (tsmom_3m,
                 "Momentum de serie temporal 3 meses (Hurst/AQR)", SALIDA_TENDENCIA),
    "tsmom_12m": (tsmom_12m,
                  "Momentum de serie temporal 12 meses (Hurst/AQR)", SALIDA_TENDENCIA),
    "tsmom_combo": (tsmom_combo,
                    "Combinacion 1/3/12 meses — la construccion de Sharpe 0.40", SALIDA_TENDENCIA),
    "tsmom_confirmado": (tsmom_confirmado,
                         "Serie temporal 3m + empuje de 7 dias (Liu & Tsyvinski)", SALIDA_TENDENCIA),
    "base_mean_reversion": (base_mean_reversion,
                            "RSI(2) de Connors — linea de base ya fallida", SALIDA_RAPIDA),
    "base_mom_transversal": (base_momentum_transversal,
                             "Momentum transversal — mi construccion erronea", SALIDA_RAPIDA),
}


def params_de(nombre: str, p: SetupParams | None = None) -> SetupParams:
    """Parametros de salida propios de cada familia."""
    p = p or SetupParams()
    return replace(p, **FAMILIAS[nombre][2])


def mascara(nombre: str, df: pd.DataFrame, ctx, p: SetupParams | None = None) -> pd.Series:
    """Mascara de entradas de una familia, con filtros minimos comunes."""
    pp = params_de(nombre, p)
    fn = FAMILIAS[nombre][0]
    f = compute_features(df, pp)
    m = fn(df, ctx, f)
    liquidez = f["dvol"] > pp.min_dollar_volume
    vol_util = f["atr_pct"].between(0.008, 0.15)   # cripto es mas volatil
    return (m & liquidez & vol_util).fillna(False).reindex(f.index).fillna(False)
