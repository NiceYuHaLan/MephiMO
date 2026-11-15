from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

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


def solve(model: LPModel, presolve: bool = True, method: str = "highs-ds", disp: bool = False) -> SolveResult:
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
    )


def compare_methods(model: LPModel, repeats: int = 3) -> list[dict]:
    import time

    rows = []
    for method in ("highs-ds", "highs-ipm"):
        times = []
        last = None
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            last = solve(model, method=method)
            times.append(time.perf_counter() - start)
        ordered = sorted(times)
        rows.append({
            "method": method,
            "status": last.status_text,
            "objective": last.fun,
            "nit": last.nit,
            "time_median": ordered[len(ordered) // 2],
            "time_min": ordered[0],
            "repeats": len(times),
        })
    return rows


def log_text(result: SolveResult, model: LPModel, title: str) -> str:
    lines = [
        f"=== {title} ===",
        f"Solver: HiGHS via scipy.optimize.linprog(method='{result.method}')",
        "LP form: min c^T z, A_ub z <= b_ub, A_eq z = b_eq, z >= 0",
        f"Variables: {len(model.c)}",
        f"Inequality rows: {model.A_ub.shape[0]}",
        f"Equality rows: {model.A_eq.shape[0]}",
        f"Status code: {result.status} ({result.status_text})",
        f"Solver message: {result.message}",
        f"Objective value: {result.fun}",
        f"Simplex/IPM iterations reported: {result.nit}",
    ]
    if result.status == 0:
        active = [
            label
            for label, row, rhs in zip(model.constraint_labels, model.A_ub, model.b_ub)
            if rhs - float(row @ result.x) <= 1e-7
        ]
        lines.append(f"Active inequality constraints ({len(active)}): {', '.join(active)}")
    else:
        lines.append("No primal solution available; see scenario diagnostics.")
    return "\n".join(lines) + "\n"
