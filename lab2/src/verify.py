import numpy as np

from . import config
from .data import Instance
from .model import LPModel, driver_flows, trip_flows
from .solve import SolveResult


def verify(inst: Instance, model: LPModel, result: SolveResult) -> dict:
    if result.status != 0 or result.x is None or result.fun is None:
        return {
            "result": result,
            "checks": [],
            "feasible": False,
            "objective_solver": None,
            "objective_numpy": None,
            "objective_diff": None,
            "x": None,
            "driver_flow": None,
            "trip_flow": None,
            "model": model,
            "instance": inst,
        }

    x = np.asarray(result.x, dtype=float)
    checks: list[dict] = []

    def check(name: str, ok: bool, violation: float) -> None:
        checks.append({"name": name, "ok": bool(ok), "violation": float(violation)})

    check(
        "границы дуг: все x >= 0",
        bool((x >= -config.TOL).all()),
        float(np.maximum(-x, 0.0).max(initial=0.0)),
    )

    cap_res = np.maximum(x - inst.arc_capacity, 0.0)
    check(
        "границы дуг: x <= ёмкость дуги",
        bool((x <= inst.arc_capacity + config.TOL).all()),
        float(cap_res.max(initial=0.0)),
    )

    driver_flow = driver_flows(inst, x)
    driver_res = np.maximum(driver_flow - inst.driver_hours, 0.0)
    check(
        "баланс вершин-водителей: приток от источника = сумма потоков по рейсам <= мощность окна",
        bool((driver_flow <= inst.driver_hours + config.TOL).all()),
        float(driver_res.max(initial=0.0)),
    )

    trip_flow = trip_flows(inst, x)
    trip_res = np.abs(trip_flow - inst.trip_hours)
    check(
        "баланс вершин-рейсов: приток = спрос, отток в сток = спрос",
        bool((trip_res <= config.TOL).all()),
        float(trip_res.max(initial=0.0)),
    )

    source_out = float(np.sum(driver_flow))
    sink_in = float(np.sum(inst.trip_hours))
    check(
        "баланс источника и стока: отток источника = приток стока",
        bool(abs(source_out - sink_in) <= config.TOL),
        abs(source_out - sink_in),
    )

    objective_numpy = float(np.dot(inst.arc_cost, x))
    objective_diff = abs(objective_numpy - float(result.fun))
    check(
        "целевая функция пересчитана из x",
        objective_diff <= 1e-6,
        objective_diff,
    )

    feasible = all(c["ok"] for c in checks)
    return {
        "result": result,
        "checks": checks,
        "feasible": bool(feasible),
        "objective_solver": float(result.fun),
        "objective_numpy": objective_numpy,
        "objective_diff": float(objective_diff),
        "x": x,
        "driver_flow": driver_flow,
        "trip_flow": trip_flow,
        "model": model,
        "instance": inst,
    }


def verify_milp(inst: Instance, y: np.ndarray, fun: float) -> dict:
    if y is None:
        return {"checks": [], "feasible": False, "objective_milp": None, "objective_numpy": None, "x": None}
    x = np.array([y[k] * inst.trip_hours[inst.arc_trip[k]] for k in range(inst.n_arcs)], dtype=float)
    checks: list[dict] = []

    def check(name: str, ok: bool, violation: float) -> None:
        checks.append({"name": name, "ok": bool(ok), "violation": float(violation)})

    check(
        "y бинарны",
        bool((np.abs(y - np.round(y)) <= config.TOL).all()),
        float(np.abs(y - np.round(y)).max(initial=0.0)),
    )
    trip_flow = trip_flows(inst, x)
    check(
        "каждый рейс назначен ровно одному водителю",
        bool((np.abs(trip_flow - inst.trip_hours) <= config.TOL).all()),
        float(np.abs(trip_flow - inst.trip_hours).max(initial=0.0)),
    )
    driver_flow = driver_flows(inst, x)
    check(
        "мощность водителей не превышена",
        bool((driver_flow <= inst.driver_hours + config.TOL).all()),
        float(np.maximum(driver_flow - inst.driver_hours, 0.0).max(initial=0.0)),
    )
    cap_res = np.maximum(x - inst.arc_capacity, 0.0)
    check(
        "ёмкости дуг соблюдены",
        bool((x <= inst.arc_capacity + config.TOL).all()),
        float(cap_res.max(initial=0.0)),
    )
    objective_numpy = float(np.dot(inst.arc_cost, x))
    diff = abs(objective_numpy - float(fun))
    check("целевая функция пересчитана", diff <= 1e-6, diff)
    return {
        "checks": checks,
        "feasible": all(c["ok"] for c in checks),
        "objective_milp": float(fun),
        "objective_numpy": objective_numpy,
        "objective_diff": diff,
        "x": x,
        "driver_flow": driver_flow,
    }


def dual_check(inst: Instance, model: LPModel, result: SolveResult, tol: float = 1e-6) -> dict:
    if result.status != 0 or result.x is None or result.ineq_marginals is None or result.eq_marginals is None:
        return {"available": False, "primal": None, "dual": None, "gap": None, "max_cs_violation": None, "rows": []}

    x = result.x
    alpha = -np.asarray(result.ineq_marginals, dtype=float)
    beta = np.asarray(result.eq_marginals, dtype=float)

    dual_obj = float(beta @ model.b_eq - alpha @ model.b_ub)
    primal_obj = float(result.fun)

    rows = []
    max_cs = 0.0
    max_rc_viol = 0.0
    slack_by_driver = {
        i: float(model.b_ub[row] - model.A_ub[row] @ x)
        for i, row in model.driver_rows.items()
    }
    for k in range(inst.n_arcs):
        i, j = inst.arc_driver[k], inst.arc_trip[k]
        d_row = model.driver_rows[i]
        a_row = model.arc_capacity_rows[k]
        gamma = float(alpha[a_row])
        rc = float(model.c[k] - beta[j] + alpha[d_row] + gamma)
        slack_driver = slack_by_driver[i]
        slack_arc = float(model.b_ub[a_row] - model.A_ub[a_row] @ x)
        cs_flow = abs(float(x[k]) * rc)
        cs_driver = abs(float(alpha[d_row]) * slack_driver)
        cs_arc = abs(gamma * slack_arc)
        rc_violation = max(0.0, -rc)
        max_cs = max(max_cs, cs_flow, cs_driver, cs_arc)
        max_rc_viol = max(max_rc_viol, rc_violation)
        rows.append({
            "arc": f"{inst.drivers[i]}→{inst.trips[j]}",
            "reduced_cost": rc,
            "flow": float(x[k]),
            "slack_driver": slack_driver,
            "slack_arc": slack_arc,
            "cs_violation": max(cs_flow, cs_driver, cs_arc),
            "rc_violation": rc_violation,
        })

    return {
        "available": True,
        "primal": primal_obj,
        "dual": dual_obj,
        "gap": abs(primal_obj - dual_obj),
        "max_cs_violation": max_cs,
        "max_rc_violation": max_rc_viol,
        "rows": rows,
        "beta": beta,
        "alpha": alpha,
    }
