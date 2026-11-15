import json

import numpy as np
import pandas as pd

from . import config
from .model import Instance


def generate(seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    P, T = len(config.PRODUCTS), len(config.WEEKS)

    demand = rng.integers(40, 121, size=(P, T))
    base_cost = rng.uniform(80.0, 160.0, size=P)
    purchase_cost = np.round(
        base_cost[:, None] * (1.0 + rng.uniform(-0.08, 0.08, size=(P, T))), 2
    )
    holding_cost = np.round(0.02 * base_cost, 2)
    initial_stock = rng.integers(10, 20, size=P)

    max_week_total = int(demand.sum(axis=0).max())
    min_capacity = int(np.ceil(initial_stock.sum() / 0.8))
    capacity = np.full(T, max(int(round(0.30 * max_week_total)), min_capacity), dtype=int)
    order_limit = np.full(T, int(round(1.50 * max_week_total)), dtype=int)
    safety_stock = np.round(0.10 * demand.mean(axis=1)).astype(int)

    return Instance(
        products=list(config.PRODUCTS),
        weeks=list(config.WEEKS),
        demand=demand,
        purchase_cost=purchase_cost,
        holding_cost=holding_cost,
        initial_stock=initial_stock,
        capacity=capacity,
        order_limit=order_limit,
        safety_stock=safety_stock,
    )


def save(inst: Instance, seed: int, data_dir, n: int = config.N, q: int = config.Q, y: int = config.Y) -> None:
    P, T = inst.n_products, inst.n_weeks

    pd.DataFrame(
        [(inst.products[p], inst.weeks[t], int(inst.demand[p, t])) for p in range(P) for t in range(T)],
        columns=["product", "week", "demand"],
    ).to_csv(data_dir / "demand.csv", index=False)

    pd.DataFrame(
        [(inst.products[p], inst.weeks[t], float(inst.purchase_cost[p, t])) for p in range(P) for t in range(T)],
        columns=["product", "week", "purchase_cost"],
    ).to_csv(data_dir / "purchase_cost.csv", index=False)

    pd.DataFrame(
        [(inst.products[p], float(inst.holding_cost[p])) for p in range(P)],
        columns=["product", "holding_cost"],
    ).to_csv(data_dir / "holding_cost.csv", index=False)

    pd.DataFrame(
        [(inst.products[p], int(inst.initial_stock[p])) for p in range(P)],
        columns=["product", "initial_stock"],
    ).to_csv(data_dir / "initial_stock.csv", index=False)

    pd.DataFrame(
        [(inst.weeks[t], int(inst.capacity[t]), int(inst.order_limit[t])) for t in range(T)],
        columns=["week", "capacity", "order_limit"],
    ).to_csv(data_dir / "capacity.csv", index=False)

    pd.DataFrame(
        [(inst.products[p], int(inst.safety_stock[p])) for p in range(P)],
        columns=["product", "safety_stock"],
    ).to_csv(data_dir / "safety_stock.csv", index=False)

    N = int(n)
    Q = int(q)
    Y = int(y)
    V = config.variant(N, Q, Y)[0]
    meta = {
        "N": N,
        "Q": Q,
        "Y": Y,
        "variant": V,
        "seed": seed,
        "products": inst.products,
        "weeks": inst.weeks,
        "units": {
            "demand": "ед./нед.",
            "purchase_cost": "руб./ед.",
            "holding_cost": "руб./(ед.·нед.)",
            "initial_stock": "ед.",
            "capacity": "ед.",
            "order_limit": "ед./нед.",
            "safety_stock": "ед.",
            "objective": f"руб. за {inst.n_weeks} нед.",
        },
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load(data_dir) -> tuple[Instance, dict]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    products = meta["products"]
    weeks = meta["weeks"]
    P, T = len(products), len(weeks)

    demand_df = pd.read_csv(data_dir / "demand.csv")
    cost_df = pd.read_csv(data_dir / "purchase_cost.csv")
    hold_df = pd.read_csv(data_dir / "holding_cost.csv")
    init_df = pd.read_csv(data_dir / "initial_stock.csv")
    cap_df = pd.read_csv(data_dir / "capacity.csv")
    safe_df = pd.read_csv(data_dir / "safety_stock.csv")

    demand = (
        demand_df.pivot(index="product", columns="week", values="demand")
        .reindex(index=products, columns=weeks)
        .to_numpy(dtype=int)
    )
    purchase_cost = (
        cost_df.pivot(index="product", columns="week", values="purchase_cost")
        .reindex(index=products, columns=weeks)
        .to_numpy(dtype=float)
    )
    holding_cost = hold_df.set_index("product").reindex(products)["holding_cost"].to_numpy(dtype=float)
    initial_stock = init_df.set_index("product").reindex(products)["initial_stock"].to_numpy(dtype=int)
    capacity = cap_df.set_index("week").reindex(weeks)["capacity"].to_numpy(dtype=int)
    order_limit = cap_df.set_index("week").reindex(weeks)["order_limit"].to_numpy(dtype=int)
    safety_stock = safe_df.set_index("product").reindex(products)["safety_stock"].to_numpy(dtype=int)

    inst = Instance(
        products=products,
        weeks=weeks,
        demand=demand,
        purchase_cost=purchase_cost,
        holding_cost=holding_cost,
        initial_stock=initial_stock,
        capacity=capacity,
        order_limit=order_limit,
        safety_stock=safety_stock,
    )
    return inst, meta


def ensure_data(seed: int, paths, regenerate: bool = False, n: int = config.N, q: int = config.Q, y: int = config.Y) -> tuple[Instance, dict]:
    meta_path = paths.data / "meta.json"
    if regenerate or not meta_path.exists():
        inst = generate(seed)
        save(inst, seed, paths.data, n=n, q=q, y=y)
    else:
        inst, meta = load(paths.data)
        if (meta.get("N"), meta.get("Q"), meta.get("Y")) != (int(n), int(q), int(y)):
            save(inst, seed, paths.data, n=n, q=q, y=y)
    return load(paths.data)
