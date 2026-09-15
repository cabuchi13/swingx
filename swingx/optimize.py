"""Optimizacion honesta de parametros: walk-forward con separacion train/test.

El problema que resuelve: si probas 200 combinaciones de parametros sobre los
mismos datos y te quedas con la mejor, esa "mejor" esta ajustada al ruido de esos
datos. Va a rendir espectacular en el backtest y mediocre en vivo. Eso es
sobreajuste, y es la causa mas comun de sistemas que funcionan hasta que se
operan con plata.

La defensa: partir el historico en dos.
  - TRAIN (los años viejos): aca se prueban todas las combinaciones y se elige.
  - TEST  (los años recientes): el sistema nunca los vio durante la eleccion.
    El resultado en TEST es la unica estimacion honesta de que esperar.

Si TRAIN da 0.45R de esperanza y TEST da 0.05R, la estrategia no funciona:
lo que encontramos fue el ruido de los años viejos.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import SetupParams, BingXCosts
from .backtest import run_backtest, stats


# Grilla por defecto. Cada eje es una decision real del setup, no un parametro
# arbitrario: cuanta sobreventa exigir, donde poner el stop, cuando tomar
# ganancia, cuanto dejar correr y cuanto esperar.
DEFAULT_GRID = {
    "rsi_max": [10.0, 15.0, 20.0],
    "atr_stop_mult": [1.2, 1.5, 2.0],
    "target_r_1": [1.0, 1.5, 2.0],
    "trail_atr_mult": [2.0, 2.5, 3.0],
    "time_stop_days": [8, 12, 20],
}

MIN_TRADES_TRAIN = 40      # menos que esto no es evidencia, es anecdota
MIN_TRADES_TEST = 15


def split_data(data: dict, train_frac: float = 0.6):
    """Parte cada serie en train/test por fecha. El corte es el mismo para todos."""
    starts, ends = [], []
    for df in data.values():
        if len(df):
            starts.append(df.index[0])
            ends.append(df.index[-1])
    if not starts:
        raise ValueError("Sin datos")
    lo, hi = min(starts), max(ends)
    cut = lo + (hi - lo) * train_frac

    train = {t: df[df.index <= cut] for t, df in data.items()}
    test = {t: df[df.index > cut] for t, df in data.items()}
    # El test necesita arrastre historico para calcular SMA200 sin quedarse corto
    warm = {}
    for t, df in data.items():
        tail = df[df.index <= cut].tail(260)
        warm[t] = pd.concat([tail, test[t]]) if len(test[t]) else test[t]
    return train, warm, cut


def grid_combos(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def evaluate(data: dict, params: SetupParams, costs: BingXCosts, risk_per_trade: float = 0.01) -> dict:
    trades = run_backtest(data, params, costs, apply_costs=True)
    s = stats(trades, risk_per_trade)
    return s


def score_config(s: dict) -> float:
    """Criterio de seleccion. Prioriza esperanza, castiga drawdown y pocas operaciones.

    No usamos retorno total: premia configuraciones que operan muchisimo y se
    comen los costos, o que tuvieron suerte en dos trades gigantes.
    """
    n = s.get("trades", 0)
    if n < MIN_TRADES_TRAIN:
        return -999.0
    exp = s.get("avg_r", 0.0)
    pf = s.get("profit_factor", 0.0)
    if not np.isfinite(pf):
        pf = 3.0
    dd = abs(s.get("max_dd_%", 0.0))
    # esperanza es lo que manda; el profit factor confirma; el drawdown penaliza
    return exp * 100 + min(pf, 3.0) * 5 - dd * 0.5


def walk_forward(data: dict, grid: dict | None = None, costs: BingXCosts | None = None,
                 train_frac: float = 0.6, risk_per_trade: float = 0.01,
                 verbose: bool = True) -> dict:
    """Corre la grilla en TRAIN, elige, y mide esa eleccion en TEST."""
    grid = grid or DEFAULT_GRID
    costs = costs or BingXCosts()
    train, test, cut = split_data(data, train_frac)

    base = SetupParams()
    combos = list(grid_combos(grid))
    if verbose:
        print(f"Corte train/test: {cut.date()}")
        print(f"Evaluando {len(combos)} configuraciones sobre TRAIN...")

    results = []
    for i, cfg in enumerate(combos, 1):
        p = replace(base, **cfg)
        s = evaluate(train, p, costs, risk_per_trade)
        s["_cfg"] = cfg
        s["_score"] = score_config(s)
        results.append(s)
        if verbose and i % 20 == 0:
            print(f"  {i}/{len(combos)}")

    valid = [r for r in results if r["_score"] > -900]
    if not valid:
        return {"error": f"Ninguna configuracion alcanzo {MIN_TRADES_TRAIN} operaciones en train.",
                "cut": str(cut.date()), "configs_probadas": len(combos)}

    valid.sort(key=lambda r: -r["_score"])
    best = valid[0]
    best_params = replace(base, **best["_cfg"])

    # Linea base: los parametros por defecto, en ambos tramos
    base_train = evaluate(train, base, costs, risk_per_trade)
    base_test = evaluate(test, base, costs, risk_per_trade)
    # La eleccion, en el tramo que nunca vio
    best_test = evaluate(test, best_params, costs, risk_per_trade)

    degradacion = None
    if best["avg_r"] != 0:
        degradacion = round((best_test.get("avg_r", 0) - best["avg_r"]) / abs(best["avg_r"]) * 100, 1)

    veredicto = _veredicto(best, best_test)

    return {
        "cut": str(cut.date()),
        "configs_probadas": len(combos),
        "configs_validas": len(valid),
        "mejor_config": best["_cfg"],
        "train_mejor": {k: v for k, v in best.items() if not k.startswith("_")},
        "test_mejor": best_test,
        "train_base": base_train,
        "test_base": base_test,
        "degradacion_pct": degradacion,
        "veredicto": veredicto,
        "top5": [{"cfg": r["_cfg"], "avg_r": r["avg_r"], "trades": r["trades"],
                  "pf": r["profit_factor"], "dd": r["max_dd_%"]} for r in valid[:5]],
    }


def _veredicto(train_s: dict, test_s: dict) -> dict:
    """Lectura honesta del resultado. Sin adornos."""
    n_test = test_s.get("trades", 0)
    exp_test = test_s.get("avg_r", 0.0)
    exp_train = train_s.get("avg_r", 0.0)
    pf_test = test_s.get("profit_factor", 0.0)

    if n_test < MIN_TRADES_TEST:
        return {"estado": "insuficiente",
                "texto": f"Solo {n_test} operaciones fuera de muestra. No alcanza para concluir nada. "
                         "Ampliá el periodo o relajá los filtros."}
    if exp_test <= 0:
        return {"estado": "no_rentable",
                "texto": f"Esperanza negativa fuera de muestra ({exp_test:+.3f}R por operación). "
                         "La estrategia no tiene edge demostrable con estos filtros. "
                         "No la operes con dinero real."}
    if exp_train > 0 and exp_test < exp_train * 0.4:
        return {"estado": "sobreajuste",
                "texto": f"En train daba {exp_train:+.3f}R y fuera de muestra {exp_test:+.3f}R. "
                         "La caída sugiere que la optimización encontró ruido, no señal. "
                         "Usá los parámetros por defecto, no los optimizados."}
    if exp_test > 0 and pf_test > 1.2:
        return {"estado": "rentable",
                "texto": f"Esperanza positiva fuera de muestra: {exp_test:+.3f}R por operación "
                         f"con profit factor {pf_test:.2f} en {n_test} operaciones. "
                         "Es evidencia razonable, no una garantía."}
    return {"estado": "marginal",
            "texto": f"Esperanza apenas positiva ({exp_test:+.3f}R, PF {pf_test:.2f}). "
                     "El margen es demasiado fino para bancar costos y slippage reales."}


def format_report(r: dict) -> str:
    if "error" in r:
        return f"ERROR: {r['error']}"
    L = []
    L.append("=" * 64)
    L.append("  VALIDACION WALK-FORWARD")
    L.append("=" * 64)
    L.append(f"  Corte train/test : {r['cut']}")
    L.append(f"  Configuraciones  : {r['configs_probadas']} probadas, {r['configs_validas']} validas")
    L.append("")
    L.append("  " + "-" * 60)
    L.append(f"  {'':<22}{'TRAIN':>12}{'TEST':>12}")
    L.append("  " + "-" * 60)

    def line(label, a, b, key, fmt="{:.3f}"):
        va = a.get(key, 0); vb = b.get(key, 0)
        L.append(f"  {label:<22}{fmt.format(va):>12}{fmt.format(vb):>12}")

    L.append("  PARAMETROS POR DEFECTO")
    line("  operaciones", r["train_base"], r["test_base"], "trades", "{:.0f}")
    line("  esperanza (R)", r["train_base"], r["test_base"], "avg_r")
    line("  win rate %", r["train_base"], r["test_base"], "win_rate_%", "{:.1f}")
    line("  profit factor", r["train_base"], r["test_base"], "profit_factor", "{:.2f}")
    line("  max drawdown %", r["train_base"], r["test_base"], "max_dd_%", "{:.1f}")
    L.append("")
    L.append("  MEJOR CONFIGURACION EN TRAIN")
    L.append(f"  {r['mejor_config']}")
    line("  operaciones", r["train_mejor"], r["test_mejor"], "trades", "{:.0f}")
    line("  esperanza (R)", r["train_mejor"], r["test_mejor"], "avg_r")
    line("  win rate %", r["train_mejor"], r["test_mejor"], "win_rate_%", "{:.1f}")
    line("  profit factor", r["train_mejor"], r["test_mejor"], "profit_factor", "{:.2f}")
    line("  max drawdown %", r["train_mejor"], r["test_mejor"], "max_dd_%", "{:.1f}")
    L.append("  " + "-" * 60)
    if r.get("degradacion_pct") is not None:
        L.append(f"  Degradacion train -> test: {r['degradacion_pct']:+.1f}%")
    L.append("")
    v = r["veredicto"]
    L.append(f"  VEREDICTO [{v['estado'].upper()}]")
    for chunk in _wrap(v["texto"], 58):
        L.append(f"  {chunk}")
    L.append("=" * 64)
    return "\n".join(L)


def _wrap(s: str, w: int):
    words, line, out = s.split(), "", []
    for x in words:
        if len(line) + len(x) + 1 > w:
            out.append(line); line = x
        else:
            line = (line + " " + x).strip()
    if line:
        out.append(line)
    return out
