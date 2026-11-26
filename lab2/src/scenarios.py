from dataclasses import dataclass

import numpy as np

from . import config
from .data import Instance, without_arcs, with_arc_capacity, with_hours
from .model import build_model, driver_flows, trip_flows
from .solve import SolveResult, solve
from .verify import verify


@dataclass
class Scenario:
    name: str
    label: str
    instance: Instance
    description: str


def most_loaded_arcs(inst: Instance, x: np.ndarray, count: int = 1) -> list[int]:
    order = np.argsort(-x)[:count]
    return [int(k) for k in order if x[k] > config.TOL]


def build_scenarios(base: Instance, base_x: np.ndarray) -> list[Scenario]:
    scenarios = [
        Scenario(
            name="base",
            label="базовый",
            instance=base,
            description="Базовый сценарий: данные варианта при фиксированном seed.",
        )
    ]

    hot = most_loaded_arcs(base, base_x, count=2)
    hot_cap = {k: float(base.arc_capacity[k] * 0.5) for k in hot[:1]}
    scenarios.append(
        Scenario(
            name="overload",
            label="перегрузка дуги",
            instance=with_arc_capacity(base, hot_cap),
            description=(
                f"Перегрузка дуги: ёмкость самой загруженной дуги "
                f"{base.drivers[base.arc_driver[hot[0]]]}→{base.trips[base.arc_trip[hot[0]]]} "
                f"снижена на 50% (со {base.arc_capacity[hot[0]]:.1f} до {hot_cap[hot[0]]:.1f} ч)."
            ),
        )
    )

    grown = np.round(base.trip_hours * 1.2, 2)
    scenarios.append(
        Scenario(
            name="demand_up",
            label="рост спроса +20%",
            instance=with_hours(base, grown),
            description=(
                "Рост спроса: длительность каждого рейса увеличена на 20% "
                f"(Σ {base.total_demand:.1f} → {float(np.sum(grown)):.1f} ч)."
            ),
        )
    )

    fail = hot[0]
    scenarios.append(
        Scenario(
            name="arc_failure",
            label="отказ дуги",
            instance=without_arcs(base, {fail}),
            description=(
                f"Отказ дуги: закрыта самая загруженная дуга "
                f"{base.drivers[base.arc_driver[fail]]}→{base.trips[base.arc_trip[fail]]} "
                f"(поток {base_x[fail]:.1f} ч)."
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="stress",
            label="стрессовый",
            instance=without_arcs(with_hours(base, grown), {fail}),
            description=(
                "Стрессовый: отказ самой загруженной дуги одновременно с ростом спроса на 20%."
            ),
        )
    )

    critical_hours = np.round(base.trip_hours * 1.5, 2)
    critical_drop = set(most_loaded_arcs(base, base_x, count=2))
    scenarios.append(
        Scenario(
            name="critical",
            label="критический",
            instance=without_arcs(with_hours(base, critical_hours), critical_drop),
            description=(
                "Критический: спрос +50% и закрыты две самые загруженные дуги "
                "(ожидается несовместимость ограничений)."
            ),
        )
    )
    return scenarios


def run_scenarios(base: Instance, base_x: np.ndarray) -> list[dict]:
    results = []
    for sc in build_scenarios(base, base_x):
        model = build_model(sc.instance)
        res = solve(model)
        ver = verify(sc.instance, model, res)
        diagnostics = diagnose(sc.instance) if res.status != 0 else []
        results.append({
            "name": sc.name,
            "label": sc.label,
            "instance": sc.instance,
            "description": sc.description,
            "model": model,
            "result": res,
            "verification": ver,
            "diagnostics": diagnostics,
        })
    return results


def diagnose(inst: Instance) -> list[str]:
    issues: list[str] = []
    supply = inst.total_supply
    demand = inst.total_demand
    if demand > supply + config.TOL:
        issues.append(
            f"глобальный дефицит: суммарный спрос {demand:.1f} ч превышает суммарную мощность "
            f"водителей {supply:.1f} ч на {demand - supply:.1f} ч"
        )

    for j in range(inst.n_trips):
        cands = [k for k in range(inst.n_arcs) if inst.arc_trip[k] == j]
        cover = float(np.sum(inst.arc_capacity[cands])) if cands else 0.0
        if cover < inst.trip_hours[j] - config.TOL:
            names = ", ".join(inst.drivers[inst.arc_driver[k]] for k in cands) or "нет допустимых"
            issues.append(
                f"{inst.trips[j]}: доступное покрытие {cover:.1f} ч < спроса {inst.trip_hours[j]:.1f} ч "
                f"(кандидаты: {names}; дуг закрыто или окна не позволяют)"
            )

    n = inst.n_trips
    for mask in range(1, 1 << n):
        subset = [j for j in range(n) if mask & (1 << j)]
        need = float(sum(inst.trip_hours[j] for j in subset))
        cap = 0.0
        for i in range(inst.n_drivers):
            arcs_i = [k for k in range(inst.n_arcs) if inst.arc_driver[k] == i]
            hours_to_subset = sum(
                min(inst.arc_capacity[k], inst.trip_hours[inst.arc_trip[k]])
                for k in arcs_i
                if inst.arc_trip[k] in subset
            )
            cap += min(inst.driver_hours[i], hours_to_subset)
        if cap < need - config.TOL:
            names = ", ".join(inst.trips[j] for j in subset)
            issues.append(
                f"условие Холла нарушено для подмножества {{{names}}}: требуется {need:.1f} ч, "
                f"доступно не более {cap:.1f} ч с учётом мощности водителей и закрытых дуг"
            )

    seen: set[str] = set()
    unique = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            unique.append(issue)
    return unique


def shadow_prices(base_result: SolveResult, model, tol: float = 1e-7) -> list[dict]:
    if base_result.status != 0 or base_result.ineq_marginals is None:
        return []
    rows = []
    for i, label in enumerate(model.constraint_labels):
        marginal = float(base_result.ineq_marginals[i])
        shadow = -marginal
        slack = float(model.b_ub[i] - model.A_ub[i] @ base_result.x)
        active = bool(slack <= tol)
        if label.startswith("мощность["):
            ctype = "driver_capacity"
        else:
            ctype = "arc_capacity"
        rows.append({
            "label": label,
            "type": ctype,
            "constraint_rhs": float(model.b_ub[i]),
            "marginal": marginal,
            "shadow_price": shadow,
            "active": active,
            "slack": slack,
        })
    return rows


def demand_duals(base_result: SolveResult, model) -> list[dict]:
    if base_result.status != 0 or base_result.eq_marginals is None:
        return []
    rows = []
    for j, label in enumerate(model.equality_labels):
        rows.append({
            "label": label,
            "type": "trip_demand",
            "rhs": float(model.b_eq[j]),
            "dual": float(base_result.eq_marginals[j]),
        })
    return rows


def shadow_prediction(shadows: list[dict], scenario_result: dict, base_fun: float) -> dict | None:
    if base_fun is None or scenario_result["result"].fun is None:
        return None
    actual = float(scenario_result["result"].fun - base_fun)
    return {"actual_delta": actual}
