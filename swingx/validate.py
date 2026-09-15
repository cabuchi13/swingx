"""Corre la validacion walk-forward y publica el resultado para el dashboard.

Pensado para GitHub Actions, que es el unico entorno con acceso a los datos
de mercado. Escribe backtest/latest.json.

Variables de entorno:
    SWINGX_PERIOD     historico a descargar (default 8y)
    SWINGX_TRAIN_FRAC fraccion para entrenamiento (default 0.6)
    SWINGX_RISK       riesgo por operacion en % (default 1.0)
"""
from __future__ import annotations

import os
import json
import datetime as dt
from pathlib import Path

from .config import UNIVERSE, SetupParams, BingXCosts
from . import data as dta
from .optimize import walk_forward, format_report, DEFAULT_GRID
from .backtest import run_backtest, stats, stats_by_bucket


def _env(name, default):
    v = os.environ.get(name, "").strip()
    return type(default)(v) if v else default


def main() -> int:
    period = _env("SWINGX_PERIOD", "8y")
    train_frac = _env("SWINGX_TRAIN_FRAC", 0.6)
    risk_pct = _env("SWINGX_RISK", 1.0)
    risk = risk_pct / 100.0
    costs = BingXCosts()

    print(f"Descargando {len(UNIVERSE)} tickers ({period})...")
    data = dta.fetch_many(UNIVERSE, period=period, use_cache=False)
    data = {t: df for t, df in data.items() if len(df) > 260}
    print(f"{len(data)} tickers con histórico suficiente\n")
    if len(data) < 5:
        print("ERROR: muy pocos tickers con datos.")
        return 1

    # 1) Validacion honesta: optimizar en train, medir en test
    r = walk_forward(data, DEFAULT_GRID, costs, train_frac=train_frac, risk_per_trade=risk)
    print(format_report(r))

    # 2) Backtest completo con los parametros por defecto, para el detalle
    print("\nBacktest completo con parámetros por defecto...")
    trades = run_backtest(data, SetupParams(), costs, apply_costs=True)
    full = stats(trades, risk)
    print(f"  {full}")

    by_score = stats_by_bucket(trades, "score")
    buckets = []
    if len(by_score):
        for idx, row in by_score.iterrows():
            buckets.append({
                "tramo": str(idx), "trades": int(row["trades"]),
                "win_rate": float(row["win_rate_%"]), "avg_r": float(row["avg_r"]),
                "profit_factor": (None if not float(row["profit_factor"]) == float(row["profit_factor"])
                                  else float(row["profit_factor"])),
            })

    por_motivo = {}
    if len(trades):
        g = trades.groupby("reason")["r"]
        for motivo, ser in g:
            por_motivo[str(motivo)] = {"n": int(ser.count()), "avg_r": round(float(ser.mean()), 3)}

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "period": period,
        "tickers": len(data),
        "walk_forward": r,
        "backtest_completo": full,
        "por_tramo_de_score": buckets,
        "por_motivo_de_salida": por_motivo,
        "peores": (trades.nsmallest(5, "r")[["ticker", "entry_date", "r", "gross_pct", "reason"]]
                   .to_dict("records") if len(trades) else []),
        "mejores": (trades.nlargest(5, "r")[["ticker", "entry_date", "r", "gross_pct", "reason"]]
                    .to_dict("records") if len(trades) else []),
    }

    out = Path("backtest")
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResultado escrito en backtest/latest.json")

    # Resumen en el summary de GitHub Actions
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        v = r.get("veredicto", {})
        with open(sm, "a") as f:
            f.write(f"## Validación walk-forward\n\n")
            f.write(f"**Veredicto: {v.get('estado', '?').upper()}**\n\n")
            f.write(f"{v.get('texto', '')}\n\n")
            f.write("```\n" + format_report(r) + "\n```\n")

    if len(trades):
        trades.to_csv(out / "trades.csv", index=False)
        print(f"Operaciones guardadas en backtest/trades.csv ({len(trades)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
