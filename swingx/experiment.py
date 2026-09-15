"""Experimento de hipotesis: cinco variantes, cada una contra su propio control.

Esto NO es una busqueda de parametros. Son cinco hipotesis declaradas de
antemano, cada una con una razon teorica, evaluadas fuera de muestra. Cinco
comparaciones no inflan el resultado como inflan 243.

Para cada variante se mide:
  - el rendimiento del SETUP con esa variante
  - el rendimiento de ENTRADAS AL AZAR bajo las MISMAS condiciones de contexto
  - la diferencia: lo unico que aporta la señal tecnica

Una variante puede subir el rendimiento absoluto sin que el setup aporte nada
(el filtro de regimen hace eso: mejora el resultado porque opera en mejores
epocas, no porque la señal sea mejor). La columna que decide es el APORTE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SetupParams, BingXCosts
from .signals import compute_features, setup_mask
from .backtest import backtest_ticker, stats
from .context import build_context, apply_context, VARIANTES


def _run_masked(data: dict, ctx: dict, p: SetupParams, costs: BingXCosts,
                cfg: dict, random_seed: int | None = None,
                densities: dict | None = None) -> pd.DataFrame:
    """Corre el backtest con la mascara del setup (o al azar) filtrada por contexto."""
    frames = []
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    for t, df in data.items():
        try:
            f = compute_features(df, p)
            if rng is None:
                m = setup_mask(f, p)
            else:
                rate = (densities or {}).get(t, 0.0)
                if rate <= 0:
                    continue
                m = pd.Series(rng.random(len(f)) < rate, index=f.index)
                m = m & ~f["sma_slow"].isna()
            m = apply_context(m, ctx.get(t), **cfg)
            if not m.any():
                continue
            tr = backtest_ticker(df, t, p, costs, apply_costs=True, armed_override=m)
            if len(tr):
                frames.append(tr)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _densities(data: dict, p: SetupParams) -> dict:
    d = {}
    for t, df in data.items():
        try:
            f = compute_features(df, p)
            m = setup_mask(f, p)
            d[t] = float(m.sum()) / max(1, len(m))
        except Exception:
            d[t] = 0.0
    return d


def run_experiment(data: dict, spy: pd.DataFrame | None, p: SetupParams | None = None,
                   costs: BingXCosts | None = None, repeats: int = 4,
                   verbose: bool = True) -> dict:
    p = p or SetupParams()
    costs = costs or BingXCosts()
    ctx = build_context(data, spy)
    dens = _densities(data, p)

    resultados = {}
    for nombre, spec in VARIANTES.items():
        cfg = spec["cfg"]
        s_setup = stats(_run_masked(data, ctx, p, costs, cfg))

        # control: mismas condiciones de contexto, entradas al azar
        ctrl = []
        for rep in range(repeats):
            tr = _run_masked(data, ctx, p, costs, cfg, random_seed=100 + rep, densities=dens)
            if len(tr):
                ctrl.append(stats(tr))
        s_ctrl = {}
        if ctrl:
            for k in ("trades", "win_rate_%", "avg_r", "profit_factor", "max_dd_%"):
                vals = [c.get(k, 0) for c in ctrl if np.isfinite(c.get(k, np.nan))]
                s_ctrl[k] = round(float(np.mean(vals)), 3) if vals else 0.0
            s_ctrl["avg_r_desvio"] = round(float(np.std([c.get("avg_r", 0) for c in ctrl])), 4)

        aporte = round(s_setup.get("avg_r", 0) - s_ctrl.get("avg_r", 0), 4)
        # ¿el aporte supera el ruido del control?
        desv = s_ctrl.get("avg_r_desvio", 0) or 0.0001
        significativo = bool(aporte > 2 * desv and s_setup.get("trades", 0) >= 30)

        resultados[nombre] = {
            "hipotesis": spec["hipotesis"],
            "setup": s_setup,
            "control": s_ctrl,
            "aporte": aporte,
            "significativo": significativo,
        }
        if verbose:
            print(f"  {nombre:26} setup {s_setup.get('avg_r',0):+.3f}R  "
                  f"azar {s_ctrl.get('avg_r',0):+.3f}R  "
                  f"aporte {aporte:+.3f}R  "
                  f"({s_setup.get('trades',0):.0f} trades)"
                  f"{'  <<< significativo' if significativo else ''}")

    ganadora = max(resultados.items(), key=lambda kv: kv[1]["aporte"])
    return {"variantes": resultados, "mejor": ganadora[0],
            "veredicto": _veredicto(resultados, ganadora)}


def _veredicto(res: dict, ganadora) -> dict:
    nombre, r = ganadora
    if not any(v["significativo"] for v in res.values()):
        return {"estado": "sin_edge",
                "texto": "Ninguna variante le gana a comprar al azar bajo las mismas "
                         "condiciones. El análisis técnico no aporta edge medible: lo que "
                         "rinde el sistema es la deriva del mercado. No operar esto con "
                         "dinero real como está."}
    if r["aporte"] < 0.05:
        return {"estado": "edge_debil",
                "texto": f"La mejor variante ('{nombre}') aporta {r['aporte']:+.3f}R sobre el azar. "
                         "Es positivo pero fino: con costos y slippage reales el margen se "
                         "puede evaporar. Sirve para papel, no para escalar capital."}
    return {"estado": "edge",
            "texto": f"La variante '{nombre}' aporta {r['aporte']:+.3f}R por operación sobre "
                     f"comprar al azar en las mismas condiciones, con "
                     f"{r['setup'].get('trades',0):.0f} operaciones. Es evidencia de que la "
                     "señal técnica aporta algo real."}


def format_report(r: dict) -> str:
    L = ["=" * 70, "  EXPERIMENTO: ¿que aporta cada capa?", "=" * 70,
         f"  {'variante':26}{'setup':>9}{'azar':>9}{'aporte':>9}{'trades':>8}", "  " + "-" * 66]
    for n, v in r["variantes"].items():
        L.append(f"  {n:26}{v['setup'].get('avg_r',0):>+9.3f}{v['control'].get('avg_r',0):>+9.3f}"
                 f"{v['aporte']:>+9.3f}{v['setup'].get('trades',0):>8.0f}"
                 + ("  *" if v["significativo"] else ""))
    L.append("  " + "-" * 66)
    L.append("  * = el aporte supera el ruido del control")
    L.append("")
    ve = r["veredicto"]
    L.append(f"  [{ve['estado'].upper()}]")
    L.append(f"  {ve['texto']}")
    L.append("=" * 70)
    return "\n".join(L)
