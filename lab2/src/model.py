from dataclasses import dataclass

import numpy as np

from .data import Instance


@dataclass
class LPModel:
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    bounds: list[tuple[float | None, float | None]]
    constraint_labels: list[str]
    equality_labels: list[str]
    arc_capacity_rows: dict[int, int]
    driver_rows: dict[int, int]

    @property
    def n_arcs(self) -> int:
        return len(self.c)


def build_model(inst: Instance) -> LPModel:
    n = inst.n_arcs
    c = inst.arc_cost.copy()

    A_eq = np.zeros((inst.n_trips, n))
    b_eq = inst.trip_hours.copy()
    equality_labels = [f"спрос[{inst.trips[j]}]" for j in range(inst.n_trips)]
    for k in range(n):
        A_eq[inst.arc_trip[k], k] = 1.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    labels: list[str] = []
    driver_rows: dict[int, int] = {}
    arc_capacity_rows: dict[int, int] = {}

    driver_hours = inst.driver_hours
    for i in range(inst.n_drivers):
        a = np.zeros(n)
        for k in range(n):
            if inst.arc_driver[k] == i:
                a[k] = 1.0
        driver_rows[i] = len(rows)
        rows.append(a)
        rhs.append(float(driver_hours[i]))
        labels.append(f"мощность[{inst.drivers[i]}]")

    for k in range(n):
        a = np.zeros(n)
        a[k] = 1.0
        arc_capacity_rows[k] = len(rows)
        rows.append(a)
        rhs.append(float(inst.arc_capacity[k]))
        labels.append(f"дуга[{inst.drivers[inst.arc_driver[k]]}→{inst.trips[inst.arc_trip[k]]}]")

    A_ub = np.vstack(rows)
    b_ub = np.asarray(rhs, dtype=float)
    bounds = [(0.0, None)] * n

    return LPModel(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        constraint_labels=labels,
        equality_labels=equality_labels,
        arc_capacity_rows=arc_capacity_rows,
        driver_rows=driver_rows,
    )


def driver_flows(inst: Instance, x: np.ndarray) -> np.ndarray:
    flows = np.zeros(inst.n_drivers)
    for k in range(inst.n_arcs):
        flows[inst.arc_driver[k]] += x[k]
    return flows


def trip_flows(inst: Instance, x: np.ndarray) -> np.ndarray:
    flows = np.zeros(inst.n_trips)
    for k in range(inst.n_arcs):
        flows[inst.arc_trip[k]] += x[k]
    return flows
