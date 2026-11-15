from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    products: list[str]
    weeks: list[int]
    demand: np.ndarray
    purchase_cost: np.ndarray
    holding_cost: np.ndarray
    initial_stock: np.ndarray
    capacity: np.ndarray
    order_limit: np.ndarray
    safety_stock: np.ndarray

    @property
    def n_products(self) -> int:
        return len(self.products)

    @property
    def n_weeks(self) -> int:
        return len(self.weeks)


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
    index_x: dict[tuple[int, int], int]
    index_i: dict[tuple[int, int], int]


def build_model(inst: Instance) -> LPModel:
    P, T = inst.n_products, inst.n_weeks
    n_x = P * T
    n_vars = 2 * n_x

    index_x = {(p, t): p * T + t for p in range(P) for t in range(T)}
    index_i = {(p, t): n_x + p * T + t for p in range(P) for t in range(T)}

    c = np.zeros(n_vars)
    for (p, t), j in index_x.items():
        c[j] = inst.purchase_cost[p, t]
    for (p, t), j in index_i.items():
        c[j] = inst.holding_cost[p]

    A_eq = np.zeros((P * T, n_vars))
    b_eq = np.zeros(P * T)
    equality_labels: list[str] = []
    row = 0
    for p in range(P):
        for t in range(T):
            A_eq[row, index_i[(p, t)]] = 1.0
            A_eq[row, index_x[(p, t)]] = -1.0
            if t == 0:
                b_eq[row] = inst.initial_stock[p] - inst.demand[p, t]
            else:
                A_eq[row, index_i[(p, t - 1)]] = -1.0
                b_eq[row] = -inst.demand[p, t]
            equality_labels.append(f"balance[{inst.products[p]}, нед.{inst.weeks[t]}]")
            row += 1

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []
    ub_labels: list[str] = []

    for t in range(T):
        a = np.zeros(n_vars)
        for p in range(P):
            a[index_i[(p, t)]] = 1.0
        ub_rows.append(a)
        ub_rhs.append(float(inst.capacity[t]))
        ub_labels.append(f"capacity[нед.{inst.weeks[t]}]")

    for t in range(T):
        a = np.zeros(n_vars)
        for p in range(P):
            a[index_x[(p, t)]] = 1.0
        ub_rows.append(a)
        ub_rhs.append(float(inst.order_limit[t]))
        ub_labels.append(f"order_limit[нед.{inst.weeks[t]}]")

    for p in range(P):
        a = np.zeros(n_vars)
        a[index_i[(p, T - 1)]] = -1.0
        ub_rows.append(a)
        ub_rhs.append(-float(inst.safety_stock[p]))
        ub_labels.append(f"safety_stock[{inst.products[p]}]")

    A_ub = np.vstack(ub_rows)
    b_ub = np.asarray(ub_rhs, dtype=float)
    bounds = [(0.0, None)] * n_vars

    return LPModel(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        constraint_labels=ub_labels,
        equality_labels=equality_labels,
        index_x=index_x,
        index_i=index_i,
    )


def unpack_solution(model: LPModel, inst: Instance, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P, T = inst.n_products, inst.n_weeks
    x = np.array([[z[model.index_x[(p, t)]] for t in range(T)] for p in range(P)])
    inventory = np.array([[z[model.index_i[(p, t)]] for t in range(T)] for p in range(P)])
    return x, inventory
