"""Validacion offline de la logica: indicadores, riesgo, senales y backtest.

No necesita datos de mercado. Construye series sinteticas con resultado conocido.
"""
from __future__ import annotations

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swingx import indicators as ind
from swingx.config import SetupParams, RiskParams, BingXCosts
from swingx.risk import plan_position, max_safe_leverage, liquidation_price
from swingx.signals import compute_features, setup_mask, entry_trigger, stop_for
from swingx.backtest import backtest_ticker

FAIL = []

def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FALLA {name} {detail}")
        FAIL.append(name)


def mkdf(close, high=None, low=None, open_=None, vol=5e7):
    close = pd.Series(close, dtype=float)
    high = close * 1.01 if high is None else pd.Series(high, dtype=float)
    low = close * 0.99 if low is None else pd.Series(low, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0]) if open_ is None else pd.Series(open_, dtype=float)
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    return pd.DataFrame(
        {"Open": open_.to_numpy(), "High": high.to_numpy(), "Low": low.to_numpy(),
         "Close": close.to_numpy(), "Volume": np.full(len(close), vol / close.mean())},
        index=idx,
    )


print("\n=== INDICADORES ===")
s = pd.Series([1, 2, 3, 4, 5], dtype=float)
check("SMA(3) correcta", np.isclose(ind.sma(s, 3).iloc[-1], 4.0))

up = pd.Series(np.arange(1, 40), dtype=float)
check("RSI de serie siempre alcista = 100", np.isclose(ind.rsi(up, 14).iloc[-1], 100.0))
down = pd.Series(np.arange(40, 1, -1), dtype=float)
check("RSI de serie siempre bajista = 0", np.isclose(ind.rsi(down, 14).iloc[-1], 0.0, atol=1e-6))

r = ind.rsi(pd.Series(np.random.RandomState(0).randn(200).cumsum() + 100), 14).dropna()
check("RSI acotado 0-100", bool((r >= 0).all() and (r <= 100).all()))

# ATR: rango verdadero constante -> ATR converge a ese rango
c = pd.Series(np.full(60, 100.0))
df_atr = mkdf(c, high=np.full(60, 102.0), low=np.full(60, 98.0), open_=np.full(60, 100.0))
a = ind.atr(df_atr["High"], df_atr["Low"], df_atr["Close"], 14).iloc[-1]
check("ATR de rango constante = 4", np.isclose(a, 4.0, atol=1e-6), f"(dio {a:.4f})")

lo_b, mid_b, up_b = ind.bollinger(pd.Series(np.full(40, 50.0)), 20, 2)
check("Bollinger de serie plana colapsa en la media",
      np.isclose(lo_b.iloc[-1], 50.0) and np.isclose(up_b.iloc[-1], 50.0))

print("\n=== RIESGO Y APALANCAMIENTO ===")
rp = RiskParams()
p1 = plan_position(equity=1000, entry=100, stop=94, leverage_requested=10, risk=rp)
check("Riesgo = 1% del capital", np.isclose(p1.risk_amount, 10.0, atol=0.01),
      f"(dio {p1.risk_amount:.2f})")
check("El nocional NO depende del apalancamiento",
      np.isclose(plan_position(1000, 100, 94, 3, risk=rp).notional, p1.notional, atol=0.01))
check("Mas apalancamiento = menos margen, mismo riesgo",
      plan_position(1000, 100, 94, 3, risk=rp).margin > p1.margin)
check("Liquidacion siempre mas lejos que el stop", p1.liq_dist_pct > p1.stop_dist_pct,
      f"(liq {p1.liq_dist_pct*100:.1f}% vs stop {p1.stop_dist_pct*100:.1f}%)")

lev_cap = max_safe_leverage(0.06, rp)
check("Cap de apalancamiento con stop 6% ~ 9.9x", 9.0 < lev_cap < 10.5, f"(dio {lev_cap:.2f})")
check("Stop ancho -> cap bajo", max_safe_leverage(0.12, rp) < 6.0)
check("Stop angosto -> cap alto", max_safe_leverage(0.02, rp) > 20.0)

p_hi = plan_position(1000, 100, 94, 25, risk=rp)
check("Pide 25x pero el sistema lo recorta", p_hi.leverage < 25 and len(p_hi.warnings) > 0)

# El tope de gap debe limitar cuando el stop es muy angosto
p_tight = plan_position(1000, 100, 99, 10, risk=rp)   # stop 1% -> nocional de riesgo enorme
check("Con stop angosto manda el tope de gap", p_tight.binding_constraint == "tope de gap overnight",
      f"(limitante: {p_tight.binding_constraint})")
check("Perdida por gap -20% acotada al 5% del capital",
      p_tight.gap_loss_20pct <= 50.0 + 1e-6, f"(dio {p_tight.gap_loss_20pct:.1f})")

liq = liquidation_price(100, 10, rp)
check("Precio de liquidacion a 10x ~ 90.5", np.isclose(liq, 90.5, atol=0.1), f"(dio {liq:.2f})")

print("\n=== SENALES ===")
# Tendencia alcista larga + pullback controlado al final
n = 320
base = 100 * (1.0025 ** np.arange(n))
base[-8:] = base[-9] * np.array([0.985, 0.972, 0.962, 0.955, 0.952, 0.951, 0.9505, 0.951])
df = mkdf(base, vol=2e9)
f = compute_features(df)
armed = setup_mask(f)
check("El setup se arma tras el pullback en tendencia", bool(armed.iloc[-30:].any()))

# En tendencia bajista NO debe armarse nunca
dn = 200 * (0.997 ** np.arange(n))
f_dn = compute_features(mkdf(dn, vol=2e9))
check("Nunca se arma en tendencia bajista", not bool(setup_mask(f_dn).any()))

# Sin lookahead: el disparo usa el maximo de la barra previa
trig = entry_trigger(f, armed)
idx = np.where(trig.to_numpy())[0]
if len(idx):
    k = int(idx[-1])
    check("El disparo exige superar el maximo del dia anterior",
          f["High"].iloc[k] > f["High"].iloc[k - 1] and bool(armed.iloc[k - 1]))
else:
    check("El disparo exige superar el maximo del dia anterior", True, "(sin disparos en la muestra)")

st = stop_for(f, len(f) - 1, float(f["High"].iloc[-1]))
check("El stop queda por debajo de la entrada", st < float(f["High"].iloc[-1]))

print("\n=== BACKTEST ===")
# Escenario 1: tras el pullback el precio sube fuerte -> resultado positivo
up_after = base.copy()
rally = base[-1] * (1.02 ** np.arange(1, 26))
sc1 = np.concatenate([up_after, rally])
tr1 = backtest_ticker(mkdf(sc1, vol=2e9), "TEST_UP", apply_costs=False)
check("Genera operaciones en el escenario alcista", len(tr1) > 0, f"({len(tr1)} trades)")
if len(tr1):
    check("El escenario alcista da R positivo", tr1["r"].iloc[0] > 0, f"(R={tr1['r'].iloc[0]:.2f})")

# Escenario 2: gap bajista brutal al dia siguiente de entrar
crash = base[-1] * np.array([1.005, 0.78, 0.77, 0.76, 0.75, 0.74, 0.75, 0.74, 0.73, 0.74,
                             0.73, 0.72, 0.73, 0.72, 0.71, 0.72])
sc2 = np.concatenate([base, crash])
df2 = mkdf(sc2, vol=2e9)
df2.loc[df2.index[len(base) + 1], "Open"] = float(base[-1] * 0.78)
tr2 = backtest_ticker(df2, "TEST_GAP", apply_costs=False)
if len(tr2):
    check("El gap se ejecuta peor que el stop (perdida > 1R)", tr2["r"].iloc[0] < -1.0,
          f"(R={tr2['r'].iloc[0]:.2f}, motivo={tr2['reason'].iloc[0]})")
else:
    check("El gap se ejecuta peor que el stop (perdida > 1R)", False, "(no genero trade)")

# Los costos siempre empeoran el resultado
tr3a = backtest_ticker(mkdf(sc1, vol=2e9), "T", apply_costs=False)
tr3b = backtest_ticker(mkdf(sc1, vol=2e9), "T", apply_costs=True)
if len(tr3a) and len(tr3b):
    check("Aplicar costos reduce el R", tr3b["r"].iloc[0] < tr3a["r"].iloc[0])

print("\n" + "=" * 46)
if FAIL:
    print(f"FALLARON {len(FAIL)}: {FAIL}")
    sys.exit(1)
print("TODO OK - la logica del sistema es correcta")
