import numpy as np

from .data import Instance


def greedy_cheapest_first(inst: Instance) -> dict:
    """Жадная эвристика: дуги в порядке возрастания стоимости, наливаем поток жадно."""
    n = inst.n_arcs
    driver_left = inst.driver_hours.copy()
    trip_left = inst.trip_hours.copy()
    x = np.zeros(n)
    order = sorted(range(n), key=lambda k: inst.arc_cost[k])

    for k in order:
        i, j = inst.arc_driver[k], inst.arc_trip[k]
        push = min(driver_left[i], trip_left[j], inst.arc_capacity[k])
        if push > 1e-12:
            x[k] = push
            driver_left[i] -= push
            trip_left[j] -= push

    covered = float(np.sum(inst.trip_hours - trip_left))
    unmet = float(np.sum(np.maximum(trip_left, 0.0)))
    cost = float(np.dot(inst.arc_cost, x))
    saturated_arcs = [k for k in range(n) if x[k] >= inst.arc_capacity[k] - 1e-9]
    return {
        "x": x,
        "cost": cost,
        "covered_h": covered,
        "unmet_h": unmet,
        "fully_feasible": bool(unmet <= 1e-7),
        "saturated_arcs": saturated_arcs,
        "driver_left": driver_left,
        "trip_left": trip_left,
    }
