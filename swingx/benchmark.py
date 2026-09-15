"""Controles para separar el edge del setup de la simple deriva del mercado.

El problema: la estrategia es solo long y las acciones de EE.UU. subieron.
Una esperanza positiva NO demuestra nada por si sola — hay que compararla
contra comprar al azar con la misma mecanica de salida.

Tres niveles, cada uno agrega una capa:
  1. AZAR       entradas aleatorias. Mide cuanto paga la pura deriva del mercado.
  2. TENDENCIA  entradas aleatorias pero solo con precio > SMA200.
                Mide cuanto agrega el filtro de tendencia.
  3. SETUP      nuestras reglas completas.
                La diferencia contra (2) es lo unico que aporta el analisis tecnico.

Si SETUP no le gana a TENDENCIA, el RSI y el pullback son decorativos.
Si TENDENCIA no le gana a AZAR, el filtro de tendencia tampoco sirve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SetupParams, BingXCosts
from .backtest import backtest_ticker, stats
from .signals import compute_features, setup_mask


def _density(data: dict, p: SetupParams) -> dict:
    """Cuantas señales por barra genera el setup en cada ticker."""
    out = {}
    for t, df in data.items():
        try:
            f = compute_features(df, p)
            m = setup_mask(f, p)
            out[t] = float(m.sum()) / max(1, len(m))
        except Exception:
            out[t] = 0.0
    return out


def _mask_random(f: pd.DataFrame, rate: float, rng, trend_only: bool = False) -> pd.Series:
    n = len(f)
    m = pd.Series(rng.random(n) < rate, index=f.index)
    warm = f["sma_slow"].isna()
    m = m & ~warm
    if trend_only:
        m = m & (f["Close"] > f["sma_slow"])
    return m.fillna(False)


def run_control(data: dict, p: SetupParams, costs: BingXCosts, mode: str,
                seed: int = 0, repeats: int = 5) -> dict:
    """Corre un control y promedia varias repeticiones para bajar la varianza."""
    dens = _density(data, p)
    rows = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        frames = []
        for t, df in data.items():
            rate = dens.get(t, 0.0)
            if rate <= 0:
                continue
            try:
                f = compute_features(df, p)
                # el control de tendencia dispara menos seguido, compensamos
                mult = 1.0 / max(0.25, (f["Close"] > f["sma_slow"]).mean()) if mode == "tendencia" else 1.0
                m = _mask_random(f, min(0.95, rate * mult), rng, trend_only=(mode == "tendencia"))
                tr = backtest_ticker(df, t, p, costs, apply_costs=True, armed_override=m)
                if len(tr):
                    frames.append(tr)
            except Exception:
                continue
        if frames:
            rows.append(stats(pd.concat(frames, ignore_index=True)))
    if not rows:
        return {"trades": 0}
    agg = {}
    for k in ("trades", "win_rate_%", "avg_r", "profit_factor", "max_dd_%"):
        vals = [r.get(k, 0) for r in rows if np.isfinite(r.get(k, np.nan))]
        agg[k] = round(float(np.mean(vals)), 3) if vals else 0.0
    agg["repeticiones"] = len(rows)
    agg["avg_r_desvio"] = round(float(np.std([r.get("avg_r", 0) for r in rows])), 4)
    return agg


def compare(data: dict, p: SetupParams | None = None, costs: BingXCosts | None = None,
            repeats: int = 5, seed: int = 0) -> dict:
    """Setup vs. los dos controles. Devuelve el aporte incremental de cada capa."""
    p = p or SetupParams()
    costs = costs or BingXCosts()

    from .backtest import run_backtest
    setup = stats(run_backtest(data, p, costs, apply_costs=True))
    azar = run_control(data, p, costs, "azar", seed, repeats)
    tend = run_control(data, p, costs, "tendencia", seed, repeats)

    ap_tend = tend.get("avg_r", 0) - azar.get("avg_r", 0)
    ap_setup = setup.get("avg_r", 0) - tend.get("avg_r", 0)

    if setup.get("trades", 0) < 30:
        ver = {"estado": "insuficiente", "texto": "Muy pocas operaciones del setup para comparar."}
    elif ap_setup <= 0:
        ver = {"estado": "sin_aporte",
               "texto": f"El setup rinde {setup.get('avg_r',0):+.3f}R y comprar al azar dentro de la "
                        f"tendencia rinde {tend.get('avg_r',0):+.3f}R. El RSI y el pullback no agregan "
                        "nada: lo que gana el sistema es la tendencia del mercado, no la señal."}
    elif ap_setup < 0.03:
        ver = {"estado": "aporte_marginal",
               "texto": f"El setup agrega apenas {ap_setup:+.3f}R sobre comprar al azar en tendencia. "
                        "Demasiado poco para justificar la complejidad y los costos."}
    else:
        ver = {"estado": "aporta",
               "texto": f"El setup agrega {ap_setup:+.3f}R por operación sobre comprar al azar dentro "
                        f"de la tendencia ({setup.get('avg_r',0):+.3f}R vs {tend.get('avg_r',0):+.3f}R). "
                        "La señal técnica aporta algo real más allá de la deriva del mercado."}

    return {"setup": setup, "control_tendencia": tend, "control_azar": azar,
            "aporte_tendencia": round(ap_tend, 4), "aporte_setup": round(ap_setup, 4),
            "veredicto_edge": ver}
