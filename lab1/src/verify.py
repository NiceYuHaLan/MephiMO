import numpy as np

from . import config
from .model import Instance, LPModel, unpack_solution


def verify(inst: Instance, model: LPModel, result) -> dict:
    if result.status != 0 or result.x is None or result.fun is None:
        return {
            "result": result,
            "checks": [],
            "feasible": False,
            "objective_solver": None,
            "objective_numpy": None,
            "objective_diff": None,
            "purchase_part": None,
            "holding_part": None,
            "x": None,
            "inventory": None,
            "model": model,
        }

    P, T = inst.n_products, inst.n_weeks
    z = result.x
    x, inventory = unpack_solution(model, inst, z)

    checks = []

    def check(name: str, ok: bool, violation: float) -> None:
        checks.append({"name": name, "ok": bool(ok), "violation": float(violation)})

    check(
        "bounds: все x >= 0",
        bool((x >= -config.TOL).all()),
        float(np.maximum(-x, 0.0).max(initial=0.0)),
    )
    check(
        "bounds: все I >= 0",
        bool((inventory >= -config.TOL).all()),
        float(np.maximum(-inventory, 0.0).max(initial=0.0)),
    )

    balance = np.zeros((P, T))
    for p in range(P):
        for t in range(T):
            prev = inst.initial_stock[p] if t == 0 else inventory[p, t - 1]
            balance[p, t] = inventory[p, t] - prev - x[p, t] + inst.demand[p, t]
    check(
        "balance: inventory = stock + procurement - demand",
        bool((np.abs(balance) <= config.TOL).all()),
        float(np.abs(balance).max(initial=0.0)),
    )

    initial_total = float(inst.initial_stock.sum())
    initial_capacity = float(inst.capacity[0])
    check(
        "capacity: начальный запас <= capacity",
        bool(initial_total <= initial_capacity + config.TOL),
        float(max(initial_total - initial_capacity, 0.0)),
    )

    capacity_used = inventory.sum(axis=0)
    capacity_res = np.maximum(capacity_used - inst.capacity, 0.0)
    check(
        "capacity: сумма запасов <= capacity",
        bool((capacity_used <= inst.capacity + config.TOL).all()),
        float(capacity_res.max()),
    )

    order_used = x.sum(axis=0)
    order_res = np.maximum(order_used - inst.order_limit, 0.0)
    check(
        "order_limit: закупки <= лимит",
        bool((order_used <= inst.order_limit + config.TOL).all()),
        float(order_res.max()),
    )

    safety_res = np.maximum(inst.safety_stock - inventory[:, T - 1], 0.0)
    check(
        "safety_stock: конечный запас >= страховой",
        bool((inventory[:, T - 1] >= inst.safety_stock - config.TOL).all()),
        float(safety_res.max(initial=0.0)),
    )

    purchase_part = float((inst.purchase_cost * x).sum())
    holding_part = float((inst.holding_cost[:, None] * inventory).sum())
    objective_numpy = purchase_part + holding_part
    objective_diff = abs(objective_numpy - float(result.fun))

    feasible = result.status == 0 and all(c["ok"] for c in checks)
    return {
        "result": result,
        "checks": checks,
        "feasible": bool(feasible),
        "objective_solver": float(result.fun),
        "objective_numpy": objective_numpy,
        "objective_diff": float(objective_diff),
        "purchase_part": purchase_part,
        "holding_part": holding_part,
        "x": x,
        "inventory": inventory,
        "model": model,
    }
