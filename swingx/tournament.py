"""Torneo de estrategias con reserva de datos en tres etapas.

El problema de probar muchas estrategias
----------------------------------------
Probar 7 familias multiplica por 7 la chance de que alguna parezca buena por
azar. La correccion no es dejar de probar: es reservar datos que la busqueda
NO puede tocar, y exigir que la ganadora funcione tambien ahi.

Las tres etapas
---------------
  ETAPA 1 — SELECCION.   Universo de desarrollo, periodo de desarrollo.
                         Corren las 7. Se rankean. Pasan las 3 mejores.
                         Aca NO se declara nada: solo se elige a quien mirar.

  ETAPA 2 — REPLICA.     Universo de validacion (acciones distintas), mismo
                         periodo. Las 3 finalistas tienen que superar 3 sigmas
                         sobre su propio control de azar.

  ETAPA 3 — RESERVA.     Periodo final, apartado desde el principio y jamas
                         usado en las etapas 1 y 2. Es la unica estimacion
                         limpia que existe. Solo se toca una vez.

Una estrategia se declara valida SOLO si pasa la 2 y la 3. Cualquier otra
combinacion es ruido con suerte.

Por que esto es distinto de lo que veniamos haciendo
----------------------------------------------------
Antes reciclabamos el mismo periodo en cada prueba nueva. Cada mirada gastaba
un poco mas del dataset sin que se notara. Aca la reserva esta apartada por
construccion: la busqueda no puede verla ni queriendo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SetupParams, BingXCosts
from .backtest import backtest_ticker, stats
from .context import build_context
from .strategies import FAMILIAS, mascara, params_de

SIGMA_MIN = 3.0
MIN_TRADES = 40
FINALISTAS = 3
RESERVA_FRAC = 0.25      # ultimo 25% del historico, intocable hasta la etapa 3


def partir_universo(data: dict, seed: int = 7):
    tickers = sorted(data.keys())
    idx = np.random.default_rng(seed).permutation(len(tickers))
    mitad = len(tickers) // 2
    dev = {tickers[i]: data[tickers[i]] for i in idx[:mitad]}
    val = {tickers[i]: data[tickers[i]] for i in idx[mitad:]}
    return dev, val


def partir_tiempo(data: dict, reserva_frac: float = RESERVA_FRAC):
    """Aparta el ultimo tramo del historico. Devuelve (busqueda, reserva, corte)."""
    ini = min(df.index[0] for df in data.values() if len(df))
    fin = max(df.index[-1] for df in data.values() if len(df))
    corte = ini + (fin - ini) * (1 - reserva_frac)
    busq = {t: df[df.index <= corte] for t, df in data.items()}
    # la reserva necesita arrastre para calcular SMA200 y ranking de 126 dias
    res = {}
    for t, df in data.items():
        cola = df[df.index <= corte].tail(300)
        post = df[df.index > corte]
        res[t] = pd.concat([cola, post]) if len(post) else post
    return busq, res, corte


def _correr(data: dict, ctx: dict, nombre: str, p: SetupParams, costs: BingXCosts,
            seed: int | None = None, densidades: dict | None = None) -> pd.DataFrame:
    # Cada familia lleva sus propios parametros de salida: las de tendencia
    # necesitan meses de recorrido, las rapidas dias. Medirlas a todas con el
    # mismo stop por tiempo fue el error de la version anterior.
    pp = params_de(nombre, p)
    frames = []
    rng = np.random.default_rng(seed) if seed is not None else None
    for t, df in data.items():
        try:
            if rng is None:
                m = mascara(nombre, df, ctx.get(t), pp)
            else:
                tasa = (densidades or {}).get(t, 0.0)
                if tasa <= 0:
                    continue
                from .signals import compute_features
                f = compute_features(df, pp)
                m = pd.Series(rng.random(len(f)) < tasa, index=f.index)
                m = m & ~f["sma_slow"].isna()
            if not m.any():
                continue
            # El control de azar usa EXACTAMENTE los mismos parametros de salida
            # que la estrategia: la unica diferencia permitida es la entrada.
            tr = backtest_ticker(df, t, pp, costs, apply_costs=True, armed_override=m)
            if len(tr):
                frames.append(tr)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _densidades(data: dict, ctx: dict, nombre: str, p: SetupParams) -> dict:
    pp = params_de(nombre, p)
    d = {}
    for t, df in data.items():
        try:
            m = mascara(nombre, df, ctx.get(t), pp)
            d[t] = float(m.sum()) / max(1, len(m))
        except Exception:
            d[t] = 0.0
    return d


def evaluar(data: dict, ctx: dict, nombre: str, p: SetupParams, costs: BingXCosts,
            repeticiones: int = 4) -> dict:
    """Una estrategia contra su propio control de entradas al azar."""
    s = stats(_correr(data, ctx, nombre, p, costs))
    dens = _densidades(data, ctx, nombre, p)
    ctrl = []
    for i in range(repeticiones):
        tr = _correr(data, ctx, nombre, p, costs, seed=900 + i, densidades=dens)
        if len(tr):
            ctrl.append(stats(tr))
    c = {}
    if ctrl:
        for k in ("trades", "win_rate_%", "avg_r", "profit_factor", "max_dd_%"):
            v = [x.get(k, 0) for x in ctrl if np.isfinite(x.get(k, np.nan))]
            c[k] = round(float(np.mean(v)), 3) if v else 0.0
        c["sigma"] = round(float(np.std([x.get("avg_r", 0) for x in ctrl])), 4)
    aporte = round(s.get("avg_r", 0) - c.get("avg_r", 0), 4)
    sigma = max(c.get("sigma", 0) or 0.0, 0.005)
    return {"estrategia": s, "control": c, "aporte": aporte,
            "sigmas": round(aporte / sigma, 2),
            "pasa": bool(aporte > SIGMA_MIN * sigma and s.get("trades", 0) >= MIN_TRADES)}


def correr_torneo(data: dict, spy, costs: BingXCosts | None = None,
                  p: SetupParams | None = None, verbose: bool = True) -> dict:
    p = p or SetupParams()
    costs = costs or BingXCosts()

    busq, reserva, corte = partir_tiempo(data)
    dev_b, val_b = partir_universo(busq)
    dev_r, val_r = partir_universo(reserva)     # misma particion (mismo seed)
    ctx_dev = build_context(dev_b, spy); ctx_val = build_context(val_b, spy)
    ctx_res = build_context({**dev_r, **val_r}, spy)

    if verbose:
        print(f"  Reserva apartada desde {corte.date()} (ultimo {RESERVA_FRAC*100:.0f}% del historico)")
        print(f"  Universo: {len(dev_b)} acciones desarrollo / {len(val_b)} validacion")
        print(f"  Familias a probar: {len(FAMILIAS)}\n")

    # ---- ETAPA 1: seleccion sobre desarrollo -------------------------------
    if verbose:
        print("  ETAPA 1 — seleccion (universo y periodo de desarrollo)")
    etapa1 = {}
    for nombre in FAMILIAS:
        r = evaluar(dev_b, ctx_dev, nombre, p, costs)
        etapa1[nombre] = r
        if verbose:
            print(f"    {nombre:22} aporte {r['aporte']:>+7.3f}  {r['sigmas']:>5.1f}σ  "
                  f"{r['estrategia'].get('trades',0):>5.0f} trades")

    ranking = sorted(etapa1.items(), key=lambda kv: -kv[1]["aporte"])
    finalistas = [n for n, r in ranking[:FINALISTAS] if r["aporte"] > 0]
    if verbose:
        print(f"\n  Finalistas: {', '.join(finalistas) if finalistas else 'ninguna con aporte positivo'}\n")

    # ---- ETAPA 2: replica en otras acciones --------------------------------
    etapa2 = {}
    if finalistas and verbose:
        print("  ETAPA 2 — replica (acciones distintas, mismo periodo)")
    for nombre in finalistas:
        r = evaluar(val_b, ctx_val, nombre, p, costs)
        etapa2[nombre] = r
        if verbose:
            print(f"    {nombre:22} aporte {r['aporte']:>+7.3f}  {r['sigmas']:>5.1f}σ  "
                  f"{'PASA' if r['pasa'] else 'no pasa'}")

    sobrevivientes = [n for n in finalistas if etapa2.get(n, {}).get("pasa")]

    # ---- ETAPA 3: la reserva, una sola vez ---------------------------------
    etapa3 = {}
    if sobrevivientes:
        if verbose:
            print(f"\n  ETAPA 3 — reserva intocada (desde {corte.date()})")
        todo_res = {**dev_r, **val_r}
        for nombre in sobrevivientes:
            r = evaluar(todo_res, ctx_res, nombre, p, costs)
            etapa3[nombre] = r
            if verbose:
                print(f"    {nombre:22} aporte {r['aporte']:>+7.3f}  {r['sigmas']:>5.1f}σ  "
                      f"{'PASA' if r['pasa'] else 'no pasa'}")
    elif verbose:
        print("\n  ETAPA 3 — no se ejecuta: ninguna estrategia llego. La reserva queda intacta")
        print("            para una proxima tanda de hipotesis.")

    ganadoras = [n for n in sobrevivientes if etapa3.get(n, {}).get("pasa")]

    return {"corte_reserva": str(corte.date()), "familias_probadas": len(FAMILIAS),
            "etapa1": etapa1, "finalistas": finalistas, "etapa2": etapa2,
            "sobrevivientes": sobrevivientes, "etapa3": etapa3, "ganadoras": ganadoras,
            "veredicto": _veredicto(ganadoras, sobrevivientes, finalistas, etapa3)}


def _veredicto(ganadoras, sobrevivientes, finalistas, etapa3) -> dict:
    if ganadoras:
        n = ganadoras[0]; r = etapa3[n]
        return {"estado": "estrategia_valida",
                "texto": f"'{n}' paso las tres etapas: superó a comprar al azar en acciones "
                         f"distintas Y en el periodo de reserva que la busqueda nunca vio "
                         f"({r['aporte']:+.3f}R, {r['sigmas']:.1f}σ, "
                         f"{r['estrategia'].get('trades',0):.0f} operaciones). "
                         "Es la evidencia mas fuerte que este diseño puede producir."}
    if sobrevivientes:
        return {"estado": "cayo_en_reserva",
                "texto": f"{', '.join(sobrevivientes)} replicó en otras acciones pero se cayó "
                         "en el periodo de reserva. Eso significa que el efecto existia en los "
                         "años de desarrollo y dejo de existir despues: o era ruido, o era un "
                         "edge real que ya se arbitro. En ninguno de los dos casos se opera."}
    if finalistas:
        return {"estado": "no_replica",
                "texto": f"Las mejores en desarrollo ({', '.join(finalistas)}) no replicaron en "
                         "acciones distintas. Es el patron clasico del sobreajuste: la estrategia "
                         "aprendio las particularidades de esas acciones, no una regla general."}
    return {"estado": "ninguna_sirve",
            "texto": "Ninguna de las familias le gana a comprar al azar ni siquiera en el "
                     "periodo de desarrollo, que es donde deberia verse mejor. La reserva "
                     "quedo intacta: se puede usar para una tanda futura de hipotesis nuevas."}


def formato(r: dict) -> str:
    L = ["=" * 78, "  TORNEO DE ESTRATEGIAS — tres etapas con reserva", "=" * 78,
         f"  {r['familias_probadas']} familias probadas · reserva desde {r['corte_reserva']}", ""]
    L.append(f"  {'ETAPA 1 (seleccion)':30}{'aporte':>10}{'sigmas':>9}{'trades':>9}")
    L.append("  " + "-" * 60)
    for n, x in sorted(r["etapa1"].items(), key=lambda kv: -kv[1]["aporte"]):
        L.append(f"  {n:30}{x['aporte']:>+10.3f}{x['sigmas']:>9.1f}"
                 f"{x['estrategia'].get('trades',0):>9.0f}")
    if r["etapa2"]:
        L.append("")
        L.append(f"  {'ETAPA 2 (otras acciones)':30}{'aporte':>10}{'sigmas':>9}")
        L.append("  " + "-" * 60)
        for n, x in r["etapa2"].items():
            L.append(f"  {n:30}{x['aporte']:>+10.3f}{x['sigmas']:>9.1f}"
                     f"   {'PASA' if x['pasa'] else 'no'}")
    if r["etapa3"]:
        L.append("")
        L.append(f"  {'ETAPA 3 (reserva intocada)':30}{'aporte':>10}{'sigmas':>9}")
        L.append("  " + "-" * 60)
        for n, x in r["etapa3"].items():
            L.append(f"  {n:30}{x['aporte']:>+10.3f}{x['sigmas']:>9.1f}"
                     f"   {'PASA' if x['pasa'] else 'no'}")
    L.append("")
    v = r["veredicto"]
    L.append(f"  [{v['estado'].upper()}]")
    for linea in _wrap(v["texto"], 74):
        L.append(f"  {linea}")
    L.append("=" * 78)
    return "\n".join(L)


def _wrap(s, w):
    out, linea = [], ""
    for p in s.split():
        if len(linea) + len(p) + 1 > w:
            out.append(linea); linea = p
        else:
            linea = (linea + " " + p).strip()
    if linea:
        out.append(linea)
    return out
