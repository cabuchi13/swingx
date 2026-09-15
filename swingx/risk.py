"""Dimensionamiento de posicion con apalancamiento.

Idea central: el apalancamiento NO define cuanto arriesgas. Lo define la distancia
al stop y el tamano del nocional. El apalancamiento solo decide cuanto margen queda
inmovilizado. Usarlo para "abrir mas grande" es la forma mas rapida de reventar.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

from .config import BingXCosts, RiskParams


@dataclass
class PositionPlan:
    entry: float
    stop: float
    target1: float
    stop_dist_pct: float
    notional: float
    margin: float
    qty: float
    leverage: float
    leverage_requested: float
    liq_price: float
    liq_dist_pct: float
    risk_amount: float
    risk_pct_equity: float
    binding_constraint: str
    gap_loss_20pct: float
    round_trip_cost_pct_margin: float
    warnings: list

    def as_dict(self) -> dict:
        return asdict(self)


def max_safe_leverage(stop_dist_pct: float, risk: RiskParams) -> float:
    """Apalancamiento maximo para que la liquidacion quede DETRAS del stop.

    La liquidacion (margen aislado) ocurre aprox. cuando la perdida alcanza
    (1/L - mmr) del nocional. Queremos que esa distancia sea `liq_buffer` veces
    mayor que la distancia al stop, para que el stop actue primero siempre.
    """
    if stop_dist_pct <= 0:
        return 1.0
    denom = stop_dist_pct * risk.liq_buffer + risk.maintenance_margin_rate
    return max(1.0, 1.0 / denom)


def liquidation_price(entry: float, leverage: float, risk: RiskParams, side: str = "long") -> float:
    """Precio de liquidacion aproximado en margen aislado."""
    move = (1.0 / leverage) - risk.maintenance_margin_rate
    return entry * (1.0 - move) if side == "long" else entry * (1.0 + move)


def holding_cost_pct_notional(days: int, costs: BingXCosts, taker_both_sides: bool = True) -> float:
    """Costo total de ida y vuelta como fraccion del NOCIONAL."""
    fee = costs.taker_fee * 2 if taker_both_sides else (costs.taker_fee + costs.maker_fee)
    funding = costs.funding_rate * costs.funding_intervals_per_day * days
    return fee + funding


def plan_position(
    equity: float,
    entry: float,
    stop: float,
    leverage_requested: float = 10.0,
    risk: RiskParams | None = None,
    costs: BingXCosts | None = None,
    expected_days: int = 8,
    target1_r: float = 1.5,
) -> PositionPlan:
    """Calcula el tamano de la posicion respetando las tres restricciones."""
    risk = risk or RiskParams()
    costs = costs or BingXCosts()
    warnings: list = []

    if stop >= entry:
        raise ValueError("El stop debe estar por debajo de la entrada (posicion long).")

    stop_dist_pct = (entry - stop) / entry

    # --- Restriccion 1: riesgo fijo por operacion -----------------------
    notional_risk = (equity * risk.risk_per_trade) / stop_dist_pct

    # --- Restriccion 2: supervivencia al gap ----------------------------
    # En horario no operativo NO se puede cerrar. El stop no protege de un gap.
    notional_gap = (equity * risk.max_gap_loss_pct) / risk.max_gap_pct

    # --- Restriccion 3: apalancamiento seguro ---------------------------
    lev_cap = min(max_safe_leverage(stop_dist_pct, risk), costs.max_leverage)
    leverage = min(leverage_requested, lev_cap)
    if leverage < leverage_requested:
        warnings.append(
            f"Apalancamiento bajado de {leverage_requested:.0f}x a {leverage:.1f}x: "
            f"con un stop de {stop_dist_pct*100:.1f}% la liquidacion quedaria demasiado cerca."
        )

    # El nocional es el minimo de las restricciones
    if notional_risk <= notional_gap:
        notional, binding = notional_risk, "riesgo por operacion"
    else:
        notional, binding = notional_gap, "tope de gap overnight"
        warnings.append(
            "El tope de gap limita el tamano: el riesgo efectivo queda por debajo del "
            f"{risk.risk_per_trade*100:.1f}% objetivo."
        )

    # --- Restriccion 4: margen comprometido ------------------------------
    margin = notional / leverage
    max_margin = equity * risk.max_margin_pct
    if margin > max_margin:
        notional = max_margin * leverage
        margin = max_margin
        binding = "tope de margen"
        warnings.append("Tamano recortado por el tope de margen comprometido.")

    qty = notional / entry
    risk_amount = notional * stop_dist_pct
    liq = liquidation_price(entry, leverage, risk)
    liq_dist_pct = (entry - liq) / entry
    target1 = entry + (entry - stop) * target1_r

    cost_pct_notional = holding_cost_pct_notional(expected_days, costs)
    cost_pct_margin = cost_pct_notional * leverage * 100.0

    if cost_pct_margin > 8.0:
        warnings.append(
            f"Costo de mantener {expected_days} dias: {cost_pct_margin:.1f}% de tu margen. "
            "A este apalancamiento el funding se come el trade; acorta el horizonte o baja el apalancamiento."
        )
    if liq_dist_pct <= stop_dist_pct:
        warnings.append("PELIGRO: la liquidacion esta antes que el stop. No tomes esta operacion asi.")

    return PositionPlan(
        entry=entry,
        stop=stop,
        target1=target1,
        stop_dist_pct=stop_dist_pct,
        notional=notional,
        margin=margin,
        qty=qty,
        leverage=leverage,
        leverage_requested=leverage_requested,
        liq_price=liq,
        liq_dist_pct=liq_dist_pct,
        risk_amount=risk_amount,
        risk_pct_equity=risk_amount / equity,
        binding_constraint=binding,
        gap_loss_20pct=notional * risk.max_gap_pct,
        round_trip_cost_pct_margin=cost_pct_margin,
        warnings=warnings,
    )


def format_plan(p: PositionPlan, ticker: str = "") -> str:
    lines = [
        f"  Entrada {p.entry:.2f} | Stop {p.stop:.2f} ({p.stop_dist_pct*100:.1f}%) | T1 {p.target1:.2f}",
        f"  Nocional ${p.notional:,.0f} | Margen ${p.margin:,.0f} | {p.qty:.3f} unidades | {p.leverage:.1f}x",
        f"  Liquidacion {p.liq_price:.2f} (a {p.liq_dist_pct*100:.1f}%) | Riesgo ${p.risk_amount:,.0f} ({p.risk_pct_equity*100:.2f}% del capital)",
        f"  Limitante: {p.binding_constraint} | Perdida si gap -20%: ${p.gap_loss_20pct:,.0f}",
        f"  Costo ida y vuelta: {p.round_trip_cost_pct_margin:.1f}% del margen",
    ]
    if ticker:
        lines.insert(0, f"{ticker}")
    for w in p.warnings:
        lines.append(f"  [!] {w}")
    return "\n".join(lines)
