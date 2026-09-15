"""Valida el armado del mensaje de Telegram y el JSON de senales, sin red."""
from __future__ import annotations

import sys, os, json, tempfile, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swingx.config import RiskParams
from swingx.risk import plan_position
from swingx.alert import build_messages, write_signals, MAX_LEN

FAIL = []

def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FALLA {name} {detail}")
        FAIL.append(name)


def fake_row(ticker, score, entry, stop, lev=10.0, equity=2000.0, rejected=None, earn=None):
    risk = RiskParams()
    plan = plan_position(equity=equity, entry=entry, stop=stop, leverage_requested=lev, risk=risk)
    return {
        "ticker": ticker, "sector": "tech", "score": score, "close": entry * 0.985,
        "trigger": entry, "stop": stop, "atr": entry * 0.028, "atr_pct": 0.028,
        "rsi2": 8.0, "rsi14": 38.0, "pullback_pct": 0.062, "pct_vs_sma50": -0.011,
        "pct_vs_sma200": 0.143, "dvol_musd": 900.0, "stop_dist_pct": (entry - stop) / entry,
        "date": "2026-09-15", "s_trend": 80, "s_depth": 90, "s_oversold": 70,
        "s_support": 80, "s_vol": 90,
        "rejected": rejected, "earnings_in_days": earn, "plan": None if rejected else plan,
    }


risk = RiskParams()

print("\n=== MENSAJE CON CANDIDATOS ===")
rows = [fake_row("NVDA", 72, 171.20, 162.05), fake_row("MSFT", 61, 402.10, 385.00)]
msgs = build_messages(rows, 2000.0, risk)
body = "\n".join(msgs)
check("Devuelve al menos un mensaje", len(msgs) >= 1)
check("Incluye los dos tickers", "NVDA" in body and "MSFT" in body)
check("Incluye el disparador de entrada", "ENTRAR sólo si supera" in body)
check("Incluye stop y objetivo", "stop" in body and "T1" in body)
check("Incluye la regla de BingX fuera de horario", "órdenes nuevas" in body)
check("Ningun mensaje supera el limite de Telegram",
      all(len(m) <= 4096 for m in msgs), f"(max {max(len(m) for m in msgs)})")

# Las etiquetas HTML tienen que estar balanceadas o Telegram rechaza el mensaje
for tag in ("b", "i", "code"):
    o = len(re.findall(rf"<{tag}>", body)); c = len(re.findall(rf"</{tag}>", body))
    check(f"Etiquetas <{tag}> balanceadas", o == c, f"({o} abiertas, {c} cerradas)")

print("\n=== MENSAJE SIN CANDIDATOS ===")
msgs0 = build_messages([], 2000.0, risk)
check("Avisa explicitamente que no hay nada", "Sin candidatos" in msgs0[0])
check("El silencio nunca se confunde con un error", len(msgs0) == 1 and len(msgs0[0]) > 30)

print("\n=== BLOQUEADOS POR EARNINGS ===")
rows_b = [fake_row("AAPL", 68, 230.0, 218.0),
          fake_row("TSLA", 70, 250.0, 236.0, rejected="reporta en 4 dias habiles", earn=4)]
body_b = "\n".join(build_messages(rows_b, 2000.0, risk))
check("Lista los bloqueados por earnings", "Bloqueados por earnings" in body_b and "TSLA" in body_b)
check("El bloqueado no trae plan de entrada",
      body_b.count("ENTRAR sólo si supera") == 1)

print("\n=== PARTIDO EN VARIOS MENSAJES ===")
many = [fake_row(f"TK{i:02d}", 60 + i % 10, 100 + i, 94 + i) for i in range(14)]
msgs_many = build_messages(many, 2000.0, risk)
check("Parte en varios mensajes cuando excede el limite", len(msgs_many) > 1,
      f"({len(msgs_many)} mensajes)")
check("Cada parte respeta el limite",
      all(len(m) <= 4096 for m in msgs_many), f"(max {max(len(m) for m in msgs_many)})")
check("No se pierde ningun ticker al partir",
      all(f"TK{i:02d}" in "\n".join(msgs_many) for i in range(14)))

print("\n=== JSON PARA LA CAPA DE RESEARCH ===")
with tempfile.TemporaryDirectory() as td:
    path = write_signals(rows_b, 2000.0, out_dir=td)
    data = json.loads(open(path).read())
    check("Escribe latest.json", os.path.basename(path) == "latest.json")
    check("Escribe tambien el archivo del dia",
          os.path.exists(os.path.join(td, data["date"] + ".json")))
    check("Incluye todos los candidatos", len(data["candidates"]) == 2)
    check("El candidato activo trae el plan serializado",
          isinstance(data["candidates"][0].get("plan"), dict))
    check("El plan trae los campos que necesita el research",
          all(k in data["candidates"][0]["plan"] for k in
              ("entry", "stop", "target1", "notional", "margin", "leverage", "liq_price")))
    check("Marca el motivo del rechazo",
          data["candidates"][1].get("rejected") is not None)
    check("Es JSON valido y reserializable", json.dumps(data) is not None)

print("\n" + "=" * 46)
if FAIL:
    print(f"FALLARON {len(FAIL)}: {FAIL}")
    sys.exit(1)
print("TODO OK - la capa de notificacion es correcta")
