"""¿Que filtro predice ganadores de verdad?

El metodo, y por que es honesto:

Para cada caracteristica de la señal (RSI, ATR, profundidad del pullback,
distancia a la media, etc.) partimos las operaciones en quintiles y medimos
la esperanza de cada uno. Si la caracteristica sirve, la esperanza deberia
subir (o bajar) de forma consistente al recorrer los quintiles.

La parte que evita autoengañarse: TODO se mide dos veces, en la primera mitad
del historico y en la segunda. Una relacion real aparece en las dos con el
mismo signo. Una relacion de ruido aparece en una y se da vuelta en la otra.

Esto NO es una busqueda de parametros: no elegimos el mejor de muchos intentos,
medimos la fuerza de cada relacion y reportamos si sobrevive la particion.
Un resultado positivo aca sigue necesitando validacion fuera de muestra antes
de operarse; lo que este diagnostico descarta es lo que ni siquiera llega ahi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_POR_QUINTIL = 12
N_PERMUTACIONES = 300
P_MAXIMO = 0.05


def _spread(s: pd.Series, r: pd.Series, q: int = 5) -> float:
    res = _quintiles(s, r, q)
    return float(res["mean"].max() - res["mean"].min()) if res is not None else 0.0


def _p_permutacion(s: pd.Series, r: pd.Series, obs: float, rng) -> float:
    """Que tan seguido el azar produce un efecto tan grande como el observado.

    Barajamos los resultados contra la caracteristica: si el efecto real no se
    distingue del que sale al barajar, la caracteristica no predice nada.
    """
    if obs <= 0:
        return 1.0
    r_np = r.to_numpy(float)
    cuenta = 0
    for _ in range(N_PERMUTACIONES):
        mezcla = pd.Series(rng.permutation(r_np), index=r.index)
        if _spread(s, mezcla) >= obs:
            cuenta += 1
    return (cuenta + 1) / (N_PERMUTACIONES + 1)


def _quintiles(s: pd.Series, r: pd.Series, q: int = 5):
    """Esperanza media por quintil de la caracteristica."""
    try:
        cortes = pd.qcut(s, q, duplicates="drop")
    except Exception:
        return None
    g = r.groupby(cortes, observed=True)
    res = g.agg(["mean", "count"])
    if (res["count"] < MIN_POR_QUINTIL).any() or len(res) < 3:
        return None
    return res


def _tendencia(res) -> float:
    """Correlacion de Spearman entre el orden del quintil y su esperanza.
    +1 = a mas valor, mejor resultado. -1 = a mas valor, peor."""
    if res is None or len(res) < 3:
        return 0.0
    y = res["mean"].to_numpy(float)
    x = np.arange(len(y), dtype=float)
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def analizar(trades: pd.DataFrame, features: list | None = None) -> dict:
    if trades is None or len(trades) < 60:
        return {"error": f"Muy pocas operaciones ({0 if trades is None else len(trades)}) "
                         "para diagnosticar filtros."}

    t = trades.copy()
    if "entry_date" in t.columns:
        t = t.sort_values("entry_date").reset_index(drop=True)
    features = features or [c for c in t.columns if c.startswith("f_")] + \
        (["score"] if "score" in t.columns else [])

    rng = np.random.default_rng(42)
    corte = len(t) // 2
    mitad1, mitad2 = t.iloc[:corte], t.iloc[corte:]

    filas = []
    for f in features:
        if f not in t.columns or t[f].isna().all():
            continue
        s_all = _quietos(t, f)
        if s_all is None:
            continue
        res_all, res_1, res_2 = s_all, _quietos(mitad1, f), _quietos(mitad2, f)
        tend_all = _tendencia(res_all)
        tend_1 = _tendencia(res_1) if res_1 is not None else 0.0
        tend_2 = _tendencia(res_2) if res_2 is not None else 0.0

        spread = float(res_all["mean"].max() - res_all["mean"].min())
        sub = t[[f, "r"]].dropna()
        pval = _p_permutacion(sub[f], sub["r"], spread, rng)
        # Para considerarlo util pedimos DOS cosas independientes:
        #   1) el efecto no se explica por azar (permutacion)
        #   2) la direccion es la misma en las dos mitades del historico
        consistente = bool(pval < P_MAXIMO
                           and tend_1 * tend_2 > 0
                           and min(abs(tend_1), abs(tend_2)) >= 0.5
                           and abs(tend_all) >= 0.5)

        filas.append({
            "filtro": f,
            "p_valor": round(pval, 4),
            "tendencia": round(tend_all, 2),
            "mitad_1": round(tend_1, 2),
            "mitad_2": round(tend_2, 2),
            "spread_R": round(spread, 3),
            "consistente": consistente,
            "quintiles": [round(float(v), 3) for v in res_all["mean"].to_numpy()],
        })

    filas.sort(key=lambda r: -abs(r["tendencia"]))
    utiles = [r for r in filas if r["consistente"]]

    if not filas:
        ver = {"estado": "sin_datos", "texto": "No se pudo evaluar ningún filtro."}
    elif not utiles:
        ver = {"estado": "ningun_filtro_sirve",
               "texto": "Ninguna característica predice el resultado de forma consistente en "
                        "las dos mitades del histórico. Las relaciones que aparecen en una mitad "
                        "se dan vuelta en la otra: eso es ruido, no señal. No existe un filtro "
                        "que separe ganadores de perdedores dentro de este setup."}
    else:
        nombres = ", ".join(r["filtro"] for r in utiles)
        ver = {"estado": "hay_filtros",
               "texto": f"Estas características mantienen el mismo signo en las dos mitades: "
                        f"{nombres}. Es un candidato a filtro real, pero todavía necesita "
                        "validarse fuera de muestra antes de operarse."}

    return {"filtros": filas, "utiles": [r["filtro"] for r in utiles], "veredicto": ver,
            "n_operaciones": int(len(t))}


def _quietos(t: pd.DataFrame, f: str):
    sub = t[[f, "r"]].dropna()
    if len(sub) < MIN_POR_QUINTIL * 3:
        return None
    return _quintiles(sub[f], sub["r"])


def format_report(d: dict) -> str:
    if "error" in d:
        return "  " + d["error"]
    L = ["=" * 78, "  ¿QUE FILTRO PREDICE GANADORES?", "=" * 78,
         f"  {d['n_operaciones']} operaciones. 'tendencia' va de -1 a +1: cuanto se ordena",
         "  el resultado al recorrer los quintiles del filtro. Cerca de 0 = no predice.",
         "",
         f"  {'filtro':18}{'tendencia':>11}{'mitad 1':>9}{'mitad 2':>9}{'spread':>8}{'p-valor':>9}  quintiles",
         "  " + "-" * 76]
    for r in d["filtros"]:
        q = " ".join(f"{v:+.2f}" for v in r["quintiles"])
        L.append(f"  {r['filtro']:18}{r['tendencia']:>+11.2f}{r['mitad_1']:>+9.2f}"
                 f"{r['mitad_2']:>+9.2f}{r['spread_R']:>8.3f}{r['p_valor']:>9.3f}  {q}"
                 + ("  *" if r["consistente"] else ""))
    L.append("  " + "-" * 74)
    L.append("  * = efecto no explicable por azar (p<0.05) Y mismo signo en las dos mitades")
    L.append("")
    v = d["veredicto"]
    L.append(f"  [{v['estado'].upper()}]")
    L.append(f"  {v['texto']}")
    L.append("=" * 78)
    return "\n".join(L)
