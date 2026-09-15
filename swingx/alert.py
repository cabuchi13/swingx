"""Capa de notificacion: corre el screener y manda las senales a Telegram.

Pensado para correr desde GitHub Actions una vez por dia, despues del cierre
de Nueva York. Ademas escribe signals/latest.json, que es lo que lee la capa
de research.

Variables de entorno:
    TELEGRAM_BOT_TOKEN   token del bot (obligatorio para enviar)
    TELEGRAM_CHAT_ID     tu chat id (obligatorio para enviar)
    SWINGX_EQUITY        capital de la cuenta en USD (default 2000)
    SWINGX_LEVERAGE      apalancamiento deseado (default 10)
    SWINGX_RISK          riesgo por operacion en % (default 1.0)
    SWINGX_MIN_SCORE     score minimo para avisar (default 55)
    SWINGX_DRY_RUN       "1" para imprimir sin enviar
"""
from __future__ import annotations

import os
import json
import html
import datetime as dt
from pathlib import Path

from .config import UNIVERSE, SetupParams, RiskParams, BingXCosts
from .screen import screen

TG_API = "https://api.telegram.org"
MAX_LEN = 3900          # el limite de Telegram es 4096; dejamos aire


def _env(name: str, default):
    v = os.environ.get(name, "").strip()
    return type(default)(v) if v else default


def fmt_money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def build_messages(rows, equity: float, risk: RiskParams) -> list:
    """Arma los mensajes en HTML de Telegram, partidos si hace falta."""
    today = dt.date.today().strftime("%d %b %Y")
    live = [r for r in rows if not r.get("rejected")]
    blocked = [r for r in rows if r.get("rejected")]

    head = f"<b>SWINGX</b> · {today}\n<i>capital {fmt_money(equity)} · riesgo {risk.risk_per_trade*100:.1f}%</i>\n"

    if not live:
        msg = head + "\n<b>Sin candidatos hoy.</b>\n"
        msg += "El setup aparece pocas veces por mes. No forzar operaciones.\n"
        if blocked:
            msg += "\n<i>Bloqueados por earnings:</i>\n"
            for r in blocked:
                msg += f"· {html.escape(r['ticker'])} — {html.escape(str(r['rejected']))}\n"
        return [msg]

    blocks = []
    for r in live:
        p = r["plan"]
        t1_pct = (p.target1 - p.entry) / p.entry * 100
        b = (
            f"\n<b>{html.escape(r['ticker'])}</b> · score {r['score']} · {html.escape(r['sector'])}\n"
            f"<code>cierre {r['close']:.2f} · RSI2 {r['rsi2']:.0f} · ATR {r['atr_pct']*100:.1f}%</code>\n"
            f"<code>pullback {r['pullback_pct']*100:.1f}% · vs SMA50 {r['pct_vs_sma50']*100:+.1f}%</code>\n"
            f"\n<b>ENTRAR sólo si supera {p.entry:.2f}</b>\n"
            f"<code>stop  {p.stop:.2f}  ({-p.stop_dist_pct*100:.1f}%)</code>\n"
            f"<code>T1    {p.target1:.2f}  (+{t1_pct:.1f}%) → cerrar 50%, stop a BE</code>\n"
            f"\n<code>nocional {fmt_money(p.notional)} · margen {fmt_money(p.margin)} · {p.leverage:.1f}x</code>\n"
            f"<code>{p.qty:.3f} unidades · liq {p.liq_price:.2f}</code>\n"
            f"<code>riesgo {fmt_money(p.risk_amount)} ({p.risk_pct_equity*100:.2f}%) · limita: {p.binding_constraint}</code>\n"
        )
        if r.get("earnings_in_days") is not None:
            b += f"<code>earnings en {r['earnings_in_days']} ruedas</code>\n"
        for w in p.warnings:
            b += f"⚠️ <i>{html.escape(w)}</i>\n"
        blocks.append(b)

    tail = ""
    if blocked:
        tail += "\n<i>Bloqueados por earnings:</i>\n"
        for r in blocked:
            tail += f"· {html.escape(r['ticker'])} — {html.escape(str(r['rejected']))}\n"
    tail += "\n<i>Cargar como orden trigger mañana en la apertura. Fuera de horario BingX no acepta órdenes nuevas.</i>"

    msgs, cur = [], head
    for b in blocks:
        if len(cur) + len(b) > MAX_LEN:
            msgs.append(cur)
            cur = ""
        cur += b
    if len(cur) + len(tail) > MAX_LEN:
        msgs.append(cur)
        cur = ""
    cur += tail
    msgs.append(cur)
    return msgs


def send_telegram(messages: list, token: str, chat_id: str) -> bool:
    import requests

    ok = True
    for m in messages:
        r = requests.post(
            f"{TG_API}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": m, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"[telegram] error {r.status_code}: {r.text[:300]}")
            ok = False
    return ok


def write_signals(rows, equity: float, out_dir: str = "signals") -> str:
    """Escribe el JSON que consume la capa de research."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "date": today,
        "equity": equity,
        "candidates": [],
    }
    for r in rows:
        item = {k: v for k, v in r.items() if k != "plan"}
        if r.get("plan") is not None:
            item["plan"] = r["plan"].as_dict()
        payload["candidates"].append(item)

    (d / "latest.json").write_text(json.dumps(payload, indent=2, default=str))
    (d / f"{today}.json").write_text(json.dumps(payload, indent=2, default=str))
    return str(d / "latest.json")


def main() -> int:
    equity = _env("SWINGX_EQUITY", 2000.0)
    leverage = _env("SWINGX_LEVERAGE", 10.0)
    risk_pct = _env("SWINGX_RISK", 1.0)
    min_score = _env("SWINGX_MIN_SCORE", 55.0)
    dry = os.environ.get("SWINGX_DRY_RUN", "") == "1"

    p = SetupParams()
    risk = RiskParams(risk_per_trade=risk_pct / 100.0)
    costs = BingXCosts()

    print(f"Escaneando {len(UNIVERSE)} tickers · capital {equity} · {leverage}x · riesgo {risk_pct}%")
    rows = screen(UNIVERSE, equity, leverage, p, risk, costs,
                  min_score=min_score, check_earnings=True,
                  period="2y", use_cache=False)

    live = [r for r in rows if not r.get("rejected")]
    print(f"Candidatos: {len(live)} activos, {len(rows) - len(live)} bloqueados por earnings")

    path = write_signals(rows, equity)
    print(f"Señales escritas en {path}")

    msgs = build_messages(rows, equity, risk)
    if dry:
        print("\n--- DRY RUN ---\n" + "\n\n".join(msgs))
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print("ERROR: falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID")
        return 1

    return 0 if send_telegram(msgs, token, chat) else 1


if __name__ == "__main__":
    raise SystemExit(main())
