"""HIPOTESIS 2: MOMENTUM TRANSVERSAL.

Declarada ANTES de ver ningun resultado.

Que dice la hipotesis
---------------------
Las acciones que vienen rindiendo mejor que sus pares tienden a seguir
haciendolo durante semanas o meses. Es la anomalia mejor documentada y la que
mejor sobrevivio decadas de escrutinio academico y de arbitraje.

Por que es distinta de la hipotesis 1
-------------------------------------
La hipotesis 1 (pullback en tendencia) compraba DEBILIDAD de corto plazo.
El experimento mostro que eso no aporta nada: el score estaba invertido, los
candidatos mas sobrevendidos rendian PEOR. Momentum compra lo contrario:
FUERZA sostenida. No es un ajuste de la hipotesis anterior, es la opuesta.

Las tres senales del experimento que apuntan aca
------------------------------------------------
1. La unica variante con aporte positivo (+0.033R) fue regimen + fuerza
   relativa + caida relativa: fuerte, en mercado alcista, sin debilidad propia.
2. El score invertido: mas sobreventa = peor resultado.
3. El filtro de regimen dio vuelta el signo del sistema entero.

Prediccion falsable
-------------------
Si la hipotesis es correcta, comprar el tercio mas fuerte del universo en
ruptura, con mercado alcista, debe rendir por encima de comprar al azar bajo
LAS MISMAS condiciones. Si el aporte no supera 3x el ruido del control, la
hipotesis queda descartada y no se ajusta: se abandona.

Por que encaja mejor con la cuenta de Cabuchi
---------------------------------------------
Horizonte de semanas en vez de dias => muchas menos operaciones => mucho menos
funding. A 10x el funding era lo que se comia la ganancia (0.6% diario del
margen). Menos operatoria tambien significa menos exposicion al gap overnight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass

from .config import SetupParams
from .signals import compute_features
from . import indicators as ind


@dataclass
class MomentumParams:
    """Parametros de momentum. Elegidos por convencion de la literatura,
    NO optimizados sobre nuestros datos."""
    rs_lookback: int = 126          # ~6 meses, el horizonte clasico de momentum
    min_rs_rank: float = 0.70       # tercio superior del universo
    breakout_len: int = 20          # ruptura de maximo de 20 ruedas
    require_regime: bool = True     # solo con el indice sobre su SMA200
    min_dollar_volume: float = 50e6
    min_atr_pct: float = 0.010
    max_atr_pct: float = 0.070

    # Gestion: mas ancha y mas larga que mean-reversion, porque la tesis
    # es que la tendencia siga, no que rebote rapido
    atr_stop_mult: float = 2.5
    target_r_1: float = 2.0
    partial_at_t1: float = 0.5
    trail_atr_mult: float = 3.0
    time_stop_days: int = 40
    swing_low_lookback: int = 20
    atr_len: int = 14


def to_setup_params(m: MomentumParams) -> SetupParams:
    """Traduce a SetupParams para reusar el motor de backtest sin cambios."""
    return SetupParams(
        atr_len=m.atr_len,
        atr_stop_mult=m.atr_stop_mult,
        swing_low_lookback=m.swing_low_lookback,
        target_r_1=m.target_r_1,
        partial_at_t1=m.partial_at_t1,
        trail_atr_mult=m.trail_atr_mult,
        time_stop_days=m.time_stop_days,
        min_dollar_volume=m.min_dollar_volume,
        min_atr_pct=m.min_atr_pct,
        max_atr_pct=m.max_atr_pct,
    )


def momentum_mask(df: pd.DataFrame, ctx: pd.DataFrame, m: MomentumParams | None = None) -> pd.Series:
    """Senal de momentum: fuerza relativa alta + ruptura + liquidez + volatilidad util.

    El filtro de regimen se aplica aparte (via apply_context) para poder medir
    su aporte por separado, igual que hicimos con la hipotesis 1.
    """
    m = m or MomentumParams()
    p = to_setup_params(m)
    f = compute_features(df, p)
    c = f["Close"]

    # Ruptura: el cierre toca el maximo de N ruedas (fuerza, no debilidad)
    hi_n = c.rolling(m.breakout_len, min_periods=m.breakout_len).max()
    breakout = c >= hi_n * 0.999

    liquidez = f["dvol"] > m.min_dollar_volume
    volatilidad = f["atr_pct"].between(m.min_atr_pct, m.max_atr_pct)
    # tendencia propia intacta
    tendencia = (c > f["sma_slow"]) & (f["sma_fast"] > f["sma_slow"])

    mask = breakout & liquidez & volatilidad & tendencia

    if ctx is not None and "rs_rank" in ctx.columns:
        rs = ctx["rs_rank"].reindex(f.index)
        mask = mask & (rs >= m.min_rs_rank).fillna(False)

    return mask.fillna(False).reindex(f.index).fillna(False)


# Variantes de la hipotesis 2, declaradas de antemano.
VARIANTES_MOMENTUM = {
    "mom_base": {
        "hipotesis": "Ruptura de 20 dias en el tercio mas fuerte del universo, "
                     "sin filtro de regimen.",
        "cfg": {},
    },
    "mom_regimen": {
        "hipotesis": "Lo mismo, pero solo con el mercado sobre su SMA200. "
                     "Momentum falla en mercados bajistas.",
        "cfg": {"use_regime": True},
    },
    "mom_regimen_elite": {
        "hipotesis": "Solo el decil superior de fuerza relativa, con régimen. "
                     "Si momentum es real, concentrar en lo mas fuerte lo amplifica.",
        "cfg": {"use_regime": True, "min_rs_rank": 0.90},
    },
}
