"""Deteccion del setup y scoring de candidatos.

Setup principal: PULLBACK EN TENDENCIA.
Compramos debilidad de corto plazo dentro de una tendencia alcista intacta,
solo cuando el precio confirma que dejo de caer. No compramos cuchillos cayendo.

Convencion anti-lookahead: todas las condiciones se evaluan con el cierre de la
barra t. La entrada ocurre en t+1 y solo si el precio supera el maximo de t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SetupParams
from . import indicators as ind


def compute_features(df: pd.DataFrame, p: SetupParams | None = None) -> pd.DataFrame:
    p = p or SetupParams()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    f = df.copy()

    f["sma_fast"] = ind.sma(c, p.sma_fast)
    f["sma_slow"] = ind.sma(c, p.sma_slow)
    f["sma20"] = ind.sma(c, 20)
    f["rsi_fast"] = ind.rsi(c, p.rsi_len)
    f["rsi14"] = ind.rsi(c, 14)
    f["atr"] = ind.atr(h, l, c, p.atr_len)
    f["atr_pct"] = f["atr"] / c
    f["bb_low"], f["bb_mid"], f["bb_up"] = ind.bollinger(c, p.bb_len, p.bb_k)
    f["adx"] = ind.adx(h, l, c, 14)
    f["dvol"] = ind.dollar_volume(c, v, 20)

    f["hi_n"] = ind.rolling_max(c, p.lookback_high)
    f["pullback"] = (f["hi_n"] - c) / f["hi_n"]
    f["swing_low"] = ind.rolling_min(l, p.swing_low_lookback)
    f["pct_vs_fast"] = (c - f["sma_fast"]) / f["sma_fast"]
    f["pct_vs_slow"] = (c - f["sma_slow"]) / f["sma_slow"]
    f["trend_spread"] = (f["sma_fast"] - f["sma_slow"]) / f["sma_slow"]
    f["down_days"] = (c.diff() < 0).astype(int).rolling(3, min_periods=3).sum()
    return f


def setup_mask(f: pd.DataFrame, p: SetupParams | None = None) -> pd.Series:
    """Condiciones de 'armado' evaluadas al cierre de la barra t."""
    p = p or SetupParams()
    c = f["Close"]

    trend = (c > f["sma_slow"]) & (f["sma_fast"] > f["sma_slow"])
    structure = f["pct_vs_fast"] > -p.max_pct_below_fast
    liquidity = f["dvol"] > p.min_dollar_volume
    volatility = f["atr_pct"].between(p.min_atr_pct, p.max_atr_pct)
    depth = f["pullback"].between(p.pullback_min, p.pullback_max)

    oversold = (
        (f["rsi_fast"] < p.rsi_max)
        | (c <= f["bb_low"])
        | (f["down_days"] >= 3)
    )

    mask = trend & structure & liquidity & volatility & depth & oversold
    if p.min_adx > 0:
        mask &= f["adx"] > p.min_adx
    return mask.fillna(False)


def entry_trigger(f: pd.DataFrame, armed: pd.Series) -> pd.Series:
    """Disparo en t+1: el precio supera el maximo de la barra de senal."""
    prev_high = f["High"].shift(1)
    armed_prev = armed.shift(1).fillna(False).astype(bool)
    return (armed_prev & (f["High"] > prev_high)).fillna(False)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def score_row(f: pd.DataFrame, i: int, p: SetupParams | None = None) -> dict:
    """Puntaje 0-100 de la calidad del candidato en la barra i."""
    p = p or SetupParams()
    r = f.iloc[i]

    # Calidad de tendencia: SMA50 sobre SMA200 y precio sobre SMA200
    s_trend = _clip01(r["trend_spread"] / 0.15) * 0.6 + _clip01(r["pct_vs_slow"] / 0.25) * 0.4

    # Profundidad del pullback: optimo entre 5% y 9%
    pb = r["pullback"]
    s_depth = _clip01(1 - abs(pb - 0.07) / 0.07)

    # Sobreventa: cuanto mas bajo el RSI(2), mejor
    s_os = _clip01((p.rsi_max - r["rsi_fast"]) / p.rsi_max) if pd.notna(r["rsi_fast"]) else 0.0
    if pd.notna(r["bb_low"]) and r["Close"] <= r["bb_low"]:
        s_os = max(s_os, 0.75)

    # Proximidad al soporte de la SMA50 (la zona donde rebota el pullback sano)
    s_support = _clip01(1 - abs(r["pct_vs_fast"]) / 0.06)

    # Volatilidad en rango util: ni muerta ni ingobernable
    ap = r["atr_pct"]
    s_vol = _clip01(1 - abs(ap - 0.025) / 0.025) if pd.notna(ap) else 0.0

    # Liquidez
    s_liq = _clip01(np.log10(max(r["dvol"], 1) / 5e7) / 1.5) if pd.notna(r["dvol"]) else 0.0

    total = (
        s_trend * 30 + s_depth * 20 + s_os * 20 + s_support * 15 + s_vol * 10 + s_liq * 5
    )
    return {
        "score": round(float(total), 1),
        "s_trend": round(s_trend * 100, 0),
        "s_depth": round(s_depth * 100, 0),
        "s_oversold": round(s_os * 100, 0),
        "s_support": round(s_support * 100, 0),
        "s_vol": round(s_vol * 100, 0),
    }


def stop_for(f: pd.DataFrame, i: int, entry: float, p: SetupParams | None = None) -> float:
    """Stop: el mas conservador entre ATR y el minimo del swing reciente."""
    p = p or SetupParams()
    r = f.iloc[i]
    by_atr = entry - p.atr_stop_mult * r["atr"]
    by_swing = r["swing_low"] - 0.20 * r["atr"]
    return float(min(by_atr, by_swing))


def scan_latest(f: pd.DataFrame, p: SetupParams | None = None) -> dict | None:
    """Evalua la ultima barra cerrada. Devuelve el candidato o None."""
    p = p or SetupParams()
    if len(f) < p.sma_slow + 5:
        return None
    armed = setup_mask(f, p)
    i = len(f) - 1
    if not bool(armed.iloc[i]):
        return None

    r = f.iloc[i]
    trigger = float(r["High"])          # entrada si manana supera este nivel
    stop = stop_for(f, i, trigger, p)
    out = {
        "date": f.index[i].date().isoformat(),
        "close": float(r["Close"]),
        "trigger": trigger,
        "stop": stop,
        "atr": float(r["atr"]),
        "atr_pct": float(r["atr_pct"]),
        "rsi2": float(r["rsi_fast"]),
        "rsi14": float(r["rsi14"]),
        "pullback_pct": float(r["pullback"]),
        "pct_vs_sma50": float(r["pct_vs_fast"]),
        "pct_vs_sma200": float(r["pct_vs_slow"]),
        "dvol_musd": float(r["dvol"]) / 1e6,
        "stop_dist_pct": (trigger - stop) / trigger,
    }
    out.update(score_row(f, i, p))
    return out
