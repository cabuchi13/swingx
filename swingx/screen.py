"""CLI del screener: busca candidatos y arma el plan de operacion.

Uso:
    python -m swingx.screen --equity 2000 --leverage 10
    python -m swingx.screen --equity 2000 --leverage 5 --min-score 55 --no-earnings-check
    python -m swingx.screen --backtest --period 5y
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import UNIVERSE, SECTOR, LOW_VOL_TIER, SetupParams, RiskParams, BingXCosts
from . import data as dta
from .signals import compute_features, scan_latest
from .risk import plan_position, format_plan
from .backtest import run_backtest, stats, stats_by_bucket


def screen(tickers, equity, leverage, p, risk, costs, min_score=50.0,
           check_earnings=True, period="2y", use_cache=True):
    rows = []
    for t in tickers:
        try:
            df = dta.fetch(t, period=period, use_cache=use_cache)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {t}: {e}", file=sys.stderr)
            continue
        try:
            f = compute_features(df, p)
            cand = scan_latest(f, p)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {t}: {e}", file=sys.stderr)
            continue
        if cand is None or cand["score"] < min_score:
            continue

        cand["ticker"] = t
        cand["sector"] = SECTOR.get(t, "?")

        # Filtro de earnings: no entramos con un reporte dentro de la ventana
        cand["earnings_in_days"] = None
        if check_earnings:
            ed = dta.next_earnings_date(t)
            days = dta.business_days_until(ed)
            cand["earnings_in_days"] = days
            cand["earnings_date"] = ed.isoformat() if ed else None
            if days is not None and days <= p.earnings_blackout_days:
                cand["rejected"] = f"reporta en {days} dias habiles"
                rows.append(cand)
                continue
        cand["rejected"] = None

        lev = leverage
        if t not in LOW_VOL_TIER and leverage > 10:
            lev = 10.0   # alta beta: nunca por encima de 10x

        plan = plan_position(
            equity=equity, entry=cand["trigger"], stop=cand["stop"],
            leverage_requested=lev, risk=risk, costs=costs,
            expected_days=p.time_stop_days // 2, target1_r=p.target_r_1,
        )
        cand["plan"] = plan
        rows.append(cand)

    rows.sort(key=lambda r: (-r["score"]))
    return rows


def print_report(rows, equity, risk):
    live = [r for r in rows if not r.get("rejected")]
    blocked = [r for r in rows if r.get("rejected")]

    print("\n" + "=" * 72)
    print(f"  CANDIDATOS SWING  |  capital ${equity:,.0f}  |  riesgo {risk.risk_per_trade*100:.1f}% por trade")
    print("=" * 72)

    if not live:
        print("\n  Sin candidatos que cumplan todas las condiciones hoy.")
        print("  Esto es normal y es una caracteristica, no un error: el setup")
        print("  aparece unas pocas veces por mes. No fuerces operaciones.")
    else:
        seen_sector = {}
        for i, r in enumerate(live, 1):
            sec = r["sector"]
            seen_sector[sec] = seen_sector.get(sec, 0) + 1
            flag = "  [tope de sector]" if seen_sector[sec] > risk.max_per_sector else ""
            print(f"\n{i}. {r['ticker']}  score {r['score']}  ({sec}){flag}")
            print(f"   Cierre {r['close']:.2f} | RSI(2) {r['rsi2']:.0f} | RSI(14) {r['rsi14']:.0f} | "
                  f"ATR {r['atr_pct']*100:.1f}%")
            print(f"   Pullback {r['pullback_pct']*100:.1f}% desde el maximo de 20d | "
                  f"vs SMA50 {r['pct_vs_sma50']*100:+.1f}% | vs SMA200 {r['pct_vs_sma200']*100:+.1f}%")
            if r.get("earnings_in_days") is not None:
                print(f"   Earnings en {r['earnings_in_days']} dias habiles ({r.get('earnings_date')})")
            print(f"   ENTRADA solo si supera {r['trigger']:.2f}")
            print(format_plan(r["plan"]))

    if blocked:
        print("\n" + "-" * 72)
        print("  Descartados por earnings (el gap del reporte no se puede stopear):")
        for r in blocked:
            print(f"   - {r['ticker']} (score {r['score']}): {r['rejected']}")

    print("\n" + "=" * 72)
    print("  Recordatorio: fuera de horario NO podes cerrar en BingX.")
    print("  El stop no te protege de un gap. Por eso el tamano esta topeado.")
    print("=" * 72 + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Screener de swing trading para BingX")
    ap.add_argument("--equity", type=float, default=2000.0, help="capital de la cuenta en USD")
    ap.add_argument("--leverage", type=float, default=10.0, help="apalancamiento deseado")
    ap.add_argument("--risk", type=float, default=1.0, help="riesgo por operacion en %%")
    ap.add_argument("--min-score", type=float, default=50.0)
    ap.add_argument("--tickers", type=str, default="", help="lista separada por comas")
    ap.add_argument("--period", type=str, default="2y")
    ap.add_argument("--no-earnings-check", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--backtest", action="store_true", help="correr backtest en lugar de screening")
    ap.add_argument("--csv", type=str, default="", help="guardar resultados en CSV")
    a = ap.parse_args(argv)

    p = SetupParams()
    risk = RiskParams(risk_per_trade=a.risk / 100.0)
    costs = BingXCosts()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()] or UNIVERSE

    if a.backtest:
        print(f"Descargando {len(tickers)} tickers ({a.period})...")
        d = dta.fetch_many(tickers, period=a.period, use_cache=not a.no_cache)
        print(f"Backtesteando {len(d)} tickers...")
        trades = run_backtest(d, p, costs)
        if not len(trades):
            print("Sin operaciones generadas.")
            return 0
        print("\n=== RESULTADO GLOBAL (costos incluidos, en multiplos de R) ===")
        for k, v in stats(trades, risk.risk_per_trade).items():
            print(f"  {k:>16}: {v}")
        print("\n=== POR TRAMO DE SCORE ===")
        print(stats_by_bucket(trades)[["trades", "win_rate_%", "avg_r", "profit_factor", "max_dd_%"]])
        print("\n=== PEORES 5 OPERACIONES ===")
        print(trades.nsmallest(5, "r")[["ticker", "entry_date", "r", "gross_pct", "reason"]].to_string(index=False))
        if a.csv:
            trades.to_csv(a.csv, index=False)
            print(f"\nOperaciones guardadas en {a.csv}")
        return 0

    rows = screen(tickers, a.equity, a.leverage, p, risk, costs,
                  min_score=a.min_score, check_earnings=not a.no_earnings_check,
                  period=a.period, use_cache=not a.no_cache)
    print_report(rows, a.equity, risk)
    if a.csv and rows:
        out = [{k: v for k, v in r.items() if k != "plan"} for r in rows]
        pd.DataFrame(out).to_csv(a.csv, index=False)
        print(f"Guardado en {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
