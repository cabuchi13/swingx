"""Corre el torneo de estrategias sobre cripto y publica el resultado.

Variables de entorno:
    SWINGX_PERIOD    historico a descargar (default 5y)
    SWINGX_ACTIVOS   "cripto" (default) o "acciones"
"""
from __future__ import annotations

import os
import json
import datetime as dt
from pathlib import Path

from .config import BingXCosts, SetupParams, UNIVERSE
from . import data as dta
from .crypto import UNIVERSO_CRYPTO, REFERENCIA_REGIMEN
from .tournament import correr_torneo, formato


def main() -> int:
    period = os.environ.get("SWINGX_PERIOD", "5y").strip() or "5y"
    activos = os.environ.get("SWINGX_ACTIVOS", "cripto").strip().lower()

    if activos == "cripto":
        universo, referencia = UNIVERSO_CRYPTO, REFERENCIA_REGIMEN
    else:
        universo, referencia = UNIVERSE, "SPY"

    print(f"Activos: {activos} | {len(universo)} instrumentos | periodo {period}\n")
    print("Descargando...")
    data = dta.fetch_many(universo, period=period, use_cache=False)
    data = {t: df for t, df in data.items() if len(df) > 300}
    print(f"{len(data)} instrumentos con historico suficiente")

    if len(data) < 10:
        print("ERROR: muy pocos instrumentos con datos.")
        return 1

    try:
        ref = dta.fetch(referencia, period=period, use_cache=False)
        print(f"Referencia de regimen {referencia}: {len(ref)} barras\n")
    except Exception as e:
        ref = None
        print(f"[aviso] sin referencia de regimen ({e})\n")

    r = correr_torneo(data, ref, BingXCosts(), SetupParams(), verbose=True)
    print()
    print(formato(r))

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "activos": activos,
        "period": period,
        "instrumentos": len(data),
        "referencia": referencia,
        "torneo": r,
    }
    out = Path("backtest")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"torneo-{activos}.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResultado en backtest/torneo-{activos}.json")

    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        v = r["veredicto"]
        with open(sm, "a") as f:
            f.write(f"## Torneo de estrategias ({activos})\n\n")
            f.write(f"**{v['estado'].upper()}** — {v['texto']}\n\n")
            f.write("```\n" + formato(r) + "\n```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
