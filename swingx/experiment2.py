"""Test de la HIPOTESIS 2 (momentum), con universo partido y vara mas alta.

Por que la vara sube: esta es la segunda familia de hipotesis que probamos
sobre el mismo periodo historico. Cada test adicional sobre los mismos datos
aumenta la chance de encontrar un falso positivo. La correccion es exigir mas.

  - hipotesis 1: aporte > 2x el ruido del control
  - hipotesis 2: aporte > 3x el ruido del control, Y en las DOS mitades
                 del universo (desarrollo y validacion)

La particion del universo es la unica fuente de datos frescos que nos queda:
los 8 años ya los usamos. Si el edge existe, tiene que aparecer en acciones
que no participaron del desarrollo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace

from .config import BingXCosts
from .backtest import backtest_ticker, stats
from .context import build_context, apply_context
from .momentum import MomentumParams, momentum_mask, to_setup_params, VARIANTES_MOMENTUM

UMBRAL_SIGMA = 3.0
MIN_TRADES = 40


def split_universe(data: dict, seed: int = 7):
    """Parte el universo en dos mitades fijas y reproducibles."""
    tickers = sorted(data.keys())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tickers))
    mitad = len(tickers) // 2
    dev = {tickers[i]: data[tickers[i]] for i in idx[:mitad]}
    val = {tickers[i]: data[tickers[i]] for i in idx[mitad:]}
    return dev, val


def _run(data: dict, ctx: dict, mp: MomentumParams, costs: BingXCosts,
         ctx_cfg: dict, random_seed: int | None = None, densities: dict | None = None):
    sp = to_setup_params(mp)
    frames = []
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    for t, df in data.items():
        try:
            if rng is None:
                m = momentum_mask(df, ctx.get(t), mp)
            else:
                rate = (densities or {}).get(t, 0.0)
                if rate <= 0:
                    continue
                from .signals import compute_features
                f = compute_features(df, sp)
                m = pd.Series(rng.random(len(f)) < rate, index=f.index)
                m = m & ~f["sma_slow"].isna()
            m = apply_context(m, ctx.get(t), **ctx_cfg)
            if not m.any():
                continue
            tr = backtest_ticker(df, t, sp, costs, apply_costs=True, armed_override=m)
            if len(tr):
                frames.append(tr)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _densities(data: dict, ctx: dict, mp: MomentumParams) -> dict:
    d = {}
    for t, df in data.items():
        try:
            m = momentum_mask(df, ctx.get(t), mp)
            d[t] = float(m.sum()) / max(1, len(m))
        except Exception:
            d[t] = 0.0
    return d


def _evaluar(data: dict, ctx: dict, mp: MomentumParams, costs: BingXCosts,
             ctx_cfg: dict, repeats: int = 4) -> dict:
    s_setup = stats(_run(data, ctx, mp, costs, ctx_cfg))
    dens = _densities(data, ctx, mp)
    ctrl = []
    for rep in range(repeats):
        tr = _run(data, ctx, mp, costs, ctx_cfg, random_seed=500 + rep, densities=dens)
        if len(tr):
            ctrl.append(stats(tr))
    s_ctrl = {}
    if ctrl:
        for k in ("trades", "win_rate_%", "avg_r", "profit_factor", "max_dd_%"):
            vals = [c.get(k, 0) for c in ctrl if np.isfinite(c.get(k, np.nan))]
            s_ctrl[k] = round(float(np.mean(vals)), 3) if vals else 0.0
        s_ctrl["sigma"] = round(float(np.std([c.get("avg_r", 0) for c in ctrl])), 4)
    aporte = round(s_setup.get("avg_r", 0) - s_ctrl.get("avg_r", 0), 4)
    sigma = max(s_ctrl.get("sigma", 0) or 0.0, 0.005)
    return {"setup": s_setup, "control": s_ctrl, "aporte": aporte,
            "sigmas": round(aporte / sigma, 2),
            "pasa": bool(aporte > UMBRAL_SIGMA * sigma and s_setup.get("trades", 0) >= MIN_TRADES)}


def run_momentum(data: dict, spy: pd.DataFrame | None, costs: BingXCosts | None = None,
                 repeats: int = 4, verbose: bool = True) -> dict:
    costs = costs or BingXCosts()
    dev, val = split_universe(data)
    ctx_dev = build_context(dev, spy)
    ctx_val = build_context(val, spy)
    if verbose:
        print(f"  universo dividido: {len(dev)} acciones para desarrollo, "
              f"{len(val)} para validación ciega\n")

    res = {}
    for nombre, spec in VARIANTES_MOMENTUM.items():
        cfg = dict(spec["cfg"])
        mp = MomentumParams()
        if "min_rs_rank" in cfg:
            mp = replace(mp, min_rs_rank=cfg.pop("min_rs_rank"))
        d = _evaluar(dev, ctx_dev, mp, costs, cfg, repeats)
        v = _evaluar(val, ctx_val, mp, costs, cfg, repeats)
        pasa_ambas = bool(d["pasa"] and v["pasa"])
        res[nombre] = {"hipotesis": spec["hipotesis"], "desarrollo": d,
                       "validacion": v, "pasa_ambas": pasa_ambas}
        if verbose:
            print(f"  {nombre:20} dev {d['setup'].get('avg_r',0):+.3f}R vs azar "
                  f"{d['control'].get('avg_r',0):+.3f}R  aporte {d['aporte']:+.3f} ({d['sigmas']:.1f}σ)")
            print(f"  {'':20} val {v['setup'].get('avg_r',0):+.3f}R vs azar "
                  f"{v['control'].get('avg_r',0):+.3f}R  aporte {v['aporte']:+.3f} ({v['sigmas']:.1f}σ)"
                  f"{'   <<< PASA' if pasa_ambas else ''}")
    return {"variantes": res, "veredicto": _veredicto(res)}


def _veredicto(res: dict) -> dict:
    ganan = [n for n, v in res.items() if v["pasa_ambas"]]
    if ganan:
        n = ganan[0]
        v = res[n]
        return {"estado": "edge_confirmado",
                "texto": f"La variante '{n}' supera a comprar al azar en las DOS mitades del "
                         f"universo, con más de {UMBRAL_SIGMA}σ de margen "
                         f"(desarrollo {v['desarrollo']['aporte']:+.3f}R, "
                         f"validación {v['validacion']['aporte']:+.3f}R). "
                         "Es la evidencia más fuerte que podemos producir con estos datos."}
    casi = [n for n, v in res.items() if v["desarrollo"]["pasa"] or v["validacion"]["pasa"]]
    if casi:
        return {"estado": "parcial",
                "texto": f"Alguna variante pasa en una mitad del universo pero no en la otra "
                         f"({', '.join(casi)}). Eso es lo que hace el ruido: aparece en una "
                         "muestra y desaparece en la siguiente. No alcanza para operar."}
    return {"estado": "sin_edge",
            "texto": "Momentum tampoco le gana a comprar al azar bajo las mismas condiciones. "
                     "Dos familias de hipótesis descartadas con el mismo control."}


def format_report(r: dict) -> str:
    L = ["=" * 74, "  HIPOTESIS 2: MOMENTUM — universo partido, vara a 3 sigma", "=" * 74,
         f"  {'variante':20}{'mitad':>14}{'setup':>9}{'azar':>9}{'aporte':>9}{'sigmas':>8}",
         "  " + "-" * 70]
    for n, v in r["variantes"].items():
        for lbl, k in (("desarrollo", "desarrollo"), ("validación", "validacion")):
            x = v[k]
            L.append(f"  {n if lbl=='desarrollo' else '':20}{lbl:>14}"
                     f"{x['setup'].get('avg_r',0):>+9.3f}{x['control'].get('avg_r',0):>+9.3f}"
                     f"{x['aporte']:>+9.3f}{x['sigmas']:>8.1f}")
        if v["pasa_ambas"]:
            L.append(f"  {'':20}  >>> PASA EN LAS DOS MITADES")
        L.append("  " + "-" * 70)
    ve = r["veredicto"]
    L.append(f"  [{ve['estado'].upper()}]")
    L.append(f"  {ve['texto']}")
    L.append("=" * 74)
    return "\n".join(L)
