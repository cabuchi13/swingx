"""Descarga y cache de datos de mercado (yfinance)."""
from __future__ import annotations

import os
import time
import datetime as dt
from typing import Iterable

import pandas as pd

CACHE_DIR = os.environ.get("SWINGX_CACHE", os.path.expanduser("~/.swingx_cache"))
COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker.replace('/', '_')}.parquet")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana MultiIndex de yfinance y deja columnas OHLCV limpias."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if any(c in set(lvl0) for c in COLS):
            df.columns = lvl0
        else:
            df.columns = df.columns.get_level_values(-1)
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en los datos descargados")
    df = df[COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna()


def fetch(ticker: str, period: str = "3y", use_cache: bool = True, max_age_hours: int = 12) -> pd.DataFrame:
    """Descarga OHLCV diario con cache en disco."""
    path = _cache_path(ticker)
    if use_cache and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h < max_age_hours:
            try:
                return pd.read_parquet(path)
            except Exception:
                pass

    import yfinance as yf

    df = yf.download(
        ticker, period=period, interval="1d",
        auto_adjust=False, progress=False, threads=False,
    )
    if df is None or len(df) == 0:
        raise ValueError(f"Sin datos para {ticker}")
    df = _normalize(df)
    try:
        df.to_parquet(path)
    except Exception:
        pass
    return df


def fetch_many(tickers: Iterable[str], period: str = "3y", use_cache: bool = True) -> dict:
    out, errors = {}, {}
    for t in tickers:
        try:
            out[t] = fetch(t, period=period, use_cache=use_cache)
        except Exception as e:  # noqa: BLE001
            errors[t] = str(e)
    if errors:
        print(f"[data] {len(errors)} tickers sin datos: {', '.join(errors)}")
    return out


def next_earnings_date(ticker: str) -> dt.date | None:
    """Proxima fecha de earnings. Devuelve None si no se puede determinar."""
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        today = pd.Timestamp.today().normalize()
        try:
            ed = tk.get_earnings_dates(limit=12)
            if ed is not None and len(ed):
                idx = pd.to_datetime(ed.index).tz_localize(None)
                future = sorted([d for d in idx if d >= today])
                if future:
                    return future[0].date()
        except Exception:
            pass
        cal = tk.calendar
        if isinstance(cal, dict):
            v = cal.get("Earnings Date")
            if isinstance(v, list) and v:
                return pd.Timestamp(v[0]).date()
            if v is not None:
                return pd.Timestamp(v).date()
    except Exception:
        return None
    return None


def business_days_until(target: dt.date | None, frm: dt.date | None = None) -> int | None:
    if target is None:
        return None
    frm = frm or dt.date.today()
    if target < frm:
        return None
    return int(len(pd.bdate_range(frm, target)) - 1)
