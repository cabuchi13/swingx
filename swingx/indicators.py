"""Indicadores tecnicos. Implementaciones Wilder donde corresponde."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Suavizado de Wilder (equivale a EMA con alpha = 1/n)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(h, l, c), n)


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder."""
    d = c.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = wilder(gain, n)
    al = wilder(loss, n)
    rs = ag / al.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # si no hubo perdidas en la ventana, RSI = 100; si no hubo ganancias, RSI = 0
    out = out.where(al != 0.0, 100.0)
    out = out.where(~((al != 0.0) & (ag == 0.0)), 0.0)
    return out


def bollinger(c: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(c, n)
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr_n = wilder(true_range(h, l, c), n)
    pdi = 100.0 * wilder(plus_dm, n) / tr_n
    mdi = 100.0 * wilder(minus_dm, n) / tr_n
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    return wilder(dx, n)


def rolling_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).max()


def rolling_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).min()


def dollar_volume(c: pd.Series, v: pd.Series, n: int = 20) -> pd.Series:
    return (c * v).rolling(n, min_periods=n).mean()
