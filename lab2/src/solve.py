from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .data import Instance
from .model import LPModel

STATUS_TEXT = {
    0: "optimal",
    1: "iteration limit",
    2: "infeasible",
    3: "unbounded",
    4: "numerical difficulties",
}


@dataclass
class SolveResult:
    status: int
    status_text: str
    message: str
    fun: float | None
    x: np.ndarray | None
    nit: int | None
    ineq_marginals: np.ndarray | None
    eq_marginals: np.ndarray | None
    method: str
    solve_time_s: float | None = None


def solve(model: LPModel, presolve: bool = True, method: str = "highs-ds", disp: bool = False) -> SolveResult:
    import time

    start = time.perf_counter()
    res = linprog(
        model.c,
        A_ub=model.A_ub,
        b_ub=model.b_ub,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=model.bounds,
        method=method,
        options={"presolve": presolve, "disp": disp},
    )
    elapsed = time.perf_counter() - start
    ineq_marginals = None
    eq_marginals = None
    if res.status == 0:
        ineq_marginals = np.asarray(res.ineqlin.marginals, dtype=float)
        eq_marginals = np.asarray(res.eqlin.marginals, dtype=float)
    return SolveResult(
        status=int(res.status),
        status_text=STATUS_TEXT.get(int(res.status), f"status {res.status}"),
        message=str(res.message),
        fun=float(res.fun) if res.fun is not None else None,
        x=np.asarray(res.x, dtype=float) if res.x is not None else None,
        nit=int(getattr(res, "nit", -1)) if getattr(res, "nit", None) is not None else None,
        ineq_marginals=ineq_marginals,
        eq_marginals=eq_marginals,
        method=method,
        solve_time_s=elapsed,
    )


@dataclass
class MilpResult:
    status: int
    status_text: str
    message: str
    fun: float | None
    y: np.ndarray | None
    mip_gap: float | None
    mip_node_count: int | None
    solve_time_s: float | None


def solve_milp(inst: Instance, presolve: bool = True, disp: bool = False) -> MilpResult:
    import time

    n = inst.n_arcs
    hours = inst.trip_hours
    c = np.array([inst.arc_cost[k] * hours[inst.arc_trip[k]] for k in range(n)], dtype=float)

    A_eq = np.zeros((inst.n_trips, n))
    for k in range(n):
        A_eq[inst.arc_trip[k], k] = 1.0
    b_eq = np.ones(inst.n_trips)

    rows = []
    rhs = []
    for i in range(inst.n_drivers):
        a = np.zeros(n)
        for k in range(n):
            if inst.arc_driver[k] == i:
                a[k] = hours[inst.arc_trip[k]]
        rows.append(a)
        rhs.append(inst.driver_hours[i])
    A_ub = np.vstack(rows)
    b_ub = np.asarray(rhs, dtype=float)

    lb = np.zeros(n)
    ub = np.ones(n)
    for k in range(n):
        if inst.arc_capacity[k] + 1e-12 < hours[inst.arc_trip[k]]:
            ub[k] = 0.0

    start = time.perf_counter()
    res = milp(
        c,
        integrality=np.ones(n),
        bounds=Bounds(lb, ub),
        constraints=[
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub),
        ],
        options={"presolve": presolve, "disp": disp},
    )
    elapsed = time.perf_counter() - start
    status_map = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "other"}
    y = np.asarray(res.x, dtype=float) if res.x is not None else None
    return MilpResult(
        status=int(res.status),
        status_text=status_map.get(int(res.status), f"status {res.status}"),
        message=str(res.message),
        fun=float(res.fun) if res.fun is not None else None,
        y=y,
        mip_gap=float(getattr(res, "mip_gap", float("nan"))) if getattr(res, "mip_gap", None) is not None else None,
        mip_node_count=int(getattr(res, "mip_node_count", -1)) if getattr(res, "mip_node_count", None) is not None else None,
        solve_time_s=elapsed,
    )


def log_text(result: SolveResult, model: LPModel, inst: Instance, title: str) -> str:
    lines = [
        f"=== {title} ===",
        "Solver: HiGHS via scipy.optimize.linprog(method='highs-ds')",
        "LP form: min c^T x, A_ub x <= b_ub, A_eq x = b_eq, x >= 0",
        f"Variables (дуги сети): {len(model.c)}",
        f"Inequality rows (мощности водителей + ёмкости дуг): {model.A_ub.shape[0]}",
        f"Equality rows (спрос рейсов): {model.A_eq.shape[0]}",
        f"Status code: {result.status} ({result.status_text})",
        f"Solver message: {result.message}",
        f"Objective value: {result.fun}",
        f"Iterations: {result.nit}",
        f"Solve time: {result.solve_time_s:.6f} s" if result.solve_time_s is not None else "Solve time: n/a",
        f"presolve: True",
    ]
    if result.status == 0 and result.x is not None:
        active = [
            label
            for label, row, rhs in zip(model.constraint_labels, model.A_ub, model.b_ub)
            if rhs - float(row @ result.x) <= 1e-7
        ]
        lines.append(f"Active inequality constraints ({len(active)}): {', '.join(active)}")
        used = [
            f"{inst.drivers[inst.arc_driver[k]]}→{inst.trips[inst.arc_trip[k]]}={result.x[k]:.3f}"
            for k in range(inst.n_arcs)
            if result.x[k] > 1e-7
        ]
        lines.append(f"Nonzero flows: {', '.join(used)}")
    else:
        lines.append("No primal solution available; see scenario diagnostics.")
    return "\n".join(lines) + "\n"
