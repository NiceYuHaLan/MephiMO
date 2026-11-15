from dataclasses import dataclass, replace

import numpy as np

from .model import Instance, LPModel, build_model
from .solve import solve
from .verify import verify


@dataclass
class Scenario:
    name: str
    instance: Instance
    description: str


def build_scenarios(base: Instance) -> list[Scenario]:
    cap_deficit = np.floor(base.capacity * 0.8).astype(int)
    demand_stress = np.ceil(base.demand * 1.2).astype(int)
    demand_critical = np.ceil(base.demand * 1.5).astype(int)
    order_critical = np.floor(base.order_limit * 0.5).astype(int)
    return [
        Scenario(name="base", instance=base, description="Базовый сценарий: данные варианта при фиксированном seed."),
        Scenario(name="deficit", instance=replace(base, capacity=cap_deficit), description="Дефицитный сценарий: вместимость склада снижена на 20%."),
        Scenario(name="stress", instance=replace(base, capacity=cap_deficit, demand=demand_stress), description="Стрессовый сценарий: вместимость склада снижена на 20% и спрос повышен на 20%."),
        Scenario(
            name="critical",
            instance=replace(base, demand=demand_critical, order_limit=order_critical),
            description="Критический сценарий: спрос повышен на 50%, лимит закупок снижен на 50% (ожидается infeasible).",
        ),
    ]


def run_scenarios(inst: Instance) -> list[dict]:
    results = []
    for sc in build_scenarios(inst):
        model = build_model(sc.instance)
        res = solve(model)
        ver = verify(sc.instance, model, res)
        diagnostics = diagnose_infeasible(sc.instance) if res.status != 0 else []
        results.append({
            "name": sc.name,
            "instance": sc.instance,
            "description": sc.description,
            "model": model,
            "result": res,
            "verification": ver,
            "diagnostics": diagnostics,
        })
    return results


def diagnose_infeasible(inst: Instance) -> list[str]:
    T = inst.n_weeks
    issues: list[str] = []
    weekly_demand = inst.demand.sum(axis=0)
    initial_total = float(inst.initial_stock.sum())
    if initial_total > float(inst.capacity[0]) + 1e-9:
        issues.append(
            f"начальный запас {initial_total:.0f} ед. превышает вместимость склада {inst.capacity[0]:.0f} ед."
        )
    for t in range(T):
        if t == 0:
            carry_in = min(initial_total, float(inst.capacity[0]))
            carry_text = f"{carry_in:.0f} ед. начального запаса"
        else:
            carry_in = float(inst.capacity[t - 1])
            carry_text = f"не более {carry_in:.0f} ед. запаса"
        available = float(inst.order_limit[t]) + carry_in
        if weekly_demand[t] > available + 1e-9:
            issues.append(
                f"нед. {inst.weeks[t]}: спрос {weekly_demand[t]:.0f} ед. превышает доступную поставку "
                f"({inst.order_limit[t]} ед. закупок + {carry_text}) = {available:.0f} ед."
            )
    total_demand = float(inst.demand.sum())
    total_need = total_demand + float(inst.safety_stock.sum())
    total_supply = float(inst.initial_stock.sum()) + float(inst.order_limit.sum())
    if total_need > total_supply:
        issues.append(
            f"совокупный дефицит: требуется {total_need:.0f} ед. (спрос + страховые), "
            f"при доступной сумме {total_supply:.0f} ед. (начальный запас + закупки за месяц)"
        )
    if float(inst.safety_stock.sum()) > float(inst.capacity[-1]):
        issues.append(
            f"страховые запасы {inst.safety_stock.sum():.0f} ед. не помещаются вместимость склада {inst.capacity[-1]} ед."
        )
    return issues


def shadow_prices(model: LPModel, base_result, tol: float = 1e-7) -> list[dict]:
    from copy import deepcopy

    if base_result.status != 0 or base_result.x is None or base_result.ineq_marginals is None:
        return []

    rows = []
    marginal = base_result.ineq_marginals
    for i, label in enumerate(model.constraint_labels):
        value = float(marginal[i])
        shadow = -value
        slack = float(model.b_ub[i] - model.A_ub[i] @ base_result.x)
        active = bool(slack <= tol)
        b2 = deepcopy(model.b_ub)
        b2[i] = b2[i] + 1.0
        model2 = deepcopy(model)
        model2.b_ub = b2
        res2 = solve(model2)
        actual_delta = float(res2.fun - base_result.fun) if res2.status == 0 and res2.fun is not None else float("nan")
        if label.startswith("capacity"):
            ctype = "capacity"
        elif label.startswith("order_limit"):
            ctype = "order_limit"
        else:
            ctype = "safety_stock"
        rows.append({
            "label": label,
            "type": ctype,
            "constraint_rhs": float(model.b_ub[i]),
            "marginal": float(value),
            "shadow_price": float(shadow),
            "active": bool(active),
            "predicted_delta_per_unit": float(-shadow),
            "actual_delta_per_unit": float(actual_delta),
            "actual_ok": bool(abs(actual_delta - (-shadow)) < 1e-4),
        })
    return rows
