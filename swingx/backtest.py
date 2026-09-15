"""Backtest del setup, en multiplos de R (independiente del apalancamiento).

Supuestos conservadores:
- La entrada se ejecuta en t+1 al superar el maximo de t. Si abre con gap por
  encima del disparador, se entra al precio de apertura (peor precio, realista).
- Si en la misma barra se tocan stop y objetivo, se asume que toco el stop primero.
- Un gap de apertura por debajo del stop se ejecuta en la apertura, no en el stop.
  Esto es exactamente lo que pasa en BingX fuera de horario: no podes cerrar.
- Los costos (fees + funding) se descuentan del resultado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SetupParams, BingXCosts
from .signals import compute_features, setup_mask, stop_for, score_row


def backtest_ticker(
    df: pd.DataFrame,
    ticker: str = "",
    p: SetupParams | None = None,
    costs: BingXCosts | None = None,
    leverage_for_costs: float = 10.0,
    apply_costs: bool = True,
    armed_override: pd.Series | None = None,
) -> pd.DataFrame:
    """Si armed_override viene dado, se usa esa mascara de señales en lugar del
    setup. Sirve para comparar el setup contra controles (entradas al azar,
    solo filtro de tendencia) con mecanica de salida y costos IDENTICOS."""
    p = p or SetupParams()
    costs = costs or BingXCosts()
    f = compute_features(df, p)
    armed = setup_mask(f, p) if armed_override is None else armed_override.reindex(f.index).fillna(False)

    o = f["Open"].to_numpy(float)
    h = f["High"].to_numpy(float)
    lo = f["Low"].to_numpy(float)
    c = f["Close"].to_numpy(float)
    atr = f["atr"].to_numpy(float)
    sma20 = f["sma20"].to_numpy(float)
    armed_np = armed.to_numpy(bool)
    dates = f.index

    trades = []
    n = len(f)
    i = 0
    while i < n - 2:
        if not armed_np[i]:
            i += 1
            continue

        trigger = h[i]
        j = i + 1
        if h[j] <= trigger:          # no confirmo, se descarta la senal
            i += 1
            continue

        entry = max(o[j], trigger)   # si abre con gap arriba, entramos peor
        stop0 = stop_for(f, i, entry, p)
        risk_per_unit = entry - stop0
        if risk_per_unit <= 0:
            i += 1
            continue

        target1 = entry + p.target_r_1 * risk_per_unit
        stop = stop0
        part_done = False
        realized_r = 0.0
        remaining = 1.0
        peak = entry
        mae = 0.0
        exit_reason, exit_price, exit_idx = None, None, None

        for k in range(j, min(j + p.time_stop_days + 1, n)):
            # 1) gap de apertura contra la posicion: se ejecuta en la apertura
            if o[k] <= stop:
                exit_price, exit_reason, exit_idx = o[k], ("gap" if k > j else "gap_dia1"), k
                break
            # 2) stop intradiario
            if lo[k] <= stop:
                exit_price, exit_reason, exit_idx = stop, "stop", k
                break
            # 3) primer objetivo: parcial y stop a breakeven
            if (not part_done) and h[k] >= target1:
                realized_r += p.partial_at_t1 * p.target_r_1
                remaining = 1.0 - p.partial_at_t1
                part_done = True
                stop = max(stop, entry)
            # 4) trailing chandelier sobre el remanente
            peak = max(peak, h[k])
            if part_done:
                stop = max(stop, peak - p.trail_atr_mult * atr[k])
            mae = min(mae, (lo[k] - entry) / entry)
            # 5) salida por tiempo
            if k == min(j + p.time_stop_days, n - 1):
                exit_price, exit_reason, exit_idx = c[k], "tiempo", k
                break
        else:
            k = min(j + p.time_stop_days, n - 1)
            exit_price, exit_reason, exit_idx = c[k], "tiempo", k

        if exit_price is None:
            i += 1
            continue

        realized_r += remaining * (exit_price - entry) / risk_per_unit
        days = max(1, exit_idx - j + 1)

        gross_pct = (exit_price - entry) / entry
        cost_pct_notional = 0.0
        if apply_costs:
            cost_pct_notional = costs.taker_fee * 2 + costs.funding_rate * costs.funding_intervals_per_day * days
            # convertir el costo a multiplos de R
            realized_r -= cost_pct_notional / ((entry - stop0) / entry)

        sc = score_row(f, i, p)
        trades.append({
            "ticker": ticker,
            "signal_date": dates[i].date(),
            "entry_date": dates[j].date(),
            "exit_date": dates[exit_idx].date(),
            "days": days,
            "entry": round(entry, 2),
            "stop": round(stop0, 2),
            "exit": round(exit_price, 2),
            "stop_dist_pct": round((entry - stop0) / entry * 100, 2),
            "r": round(realized_r, 3),
            "gross_pct": round(gross_pct * 100, 2),
            "mae_pct": round(mae * 100, 2),
            "reason": exit_reason,
            "score": sc["score"],
            "cost_pct_margin": round(cost_pct_notional * leverage_for_costs * 100, 2),
        })

        i = exit_idx + 1   # sin posiciones solapadas en el mismo ticker

    return pd.DataFrame(trades)


def run_backtest(data: dict, p: SetupParams | None = None, costs: BingXCosts | None = None,
                 apply_costs: bool = True) -> pd.DataFrame:
    frames = []
    for t, df in data.items():
        try:
            tr = backtest_ticker(df, t, p, costs, apply_costs=apply_costs)
            if len(tr):
                frames.append(tr)
        except Exception as e:  # noqa: BLE001
            print(f"[backtest] {t}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("entry_date").reset_index(drop=True)


def stats(trades: pd.DataFrame, risk_per_trade: float = 0.01) -> dict:
    if trades is None or len(trades) == 0:
        return {"trades": 0}
    r = trades["r"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    eq = (1 + risk_per_trade * r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "trades": int(len(r)),
        "win_rate_%": round(float((r > 0).mean() * 100), 1),
        "avg_r": round(float(r.mean()), 3),
        "median_r": round(float(r.median()), 3),
        "expectancy_r": round(float(r.mean()), 3),
        "avg_win_r": round(float(wins.mean()) if len(wins) else 0.0, 2),
        "avg_loss_r": round(float(losses.mean()) if len(losses) else 0.0, 2),
        "profit_factor": round(float(gross_win / gross_loss) if gross_loss > 0 else np.inf, 2),
        "worst_r": round(float(r.min()), 2),
        "best_r": round(float(r.max()), 2),
        "avg_days": round(float(trades["days"].mean()), 1),
        "gap_exits_%": round(float(trades["reason"].str.startswith("gap").mean() * 100), 1),
        "total_R": round(float(r.sum()), 1),
        "equity_x": round(float(eq.iloc[-1]), 3),
        "max_dd_%": round(float(dd * 100), 1),
    }


def stats_by_bucket(trades: pd.DataFrame, col: str = "score", bins=(0, 40, 55, 70, 100)) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    t["bucket"] = pd.cut(t[col], bins=list(bins))
    rows = []
    for b, g in t.groupby("bucket", observed=True):
        s = stats(g)
        s["bucket"] = str(b)
        rows.append(s)
    return pd.DataFrame(rows).set_index("bucket")
