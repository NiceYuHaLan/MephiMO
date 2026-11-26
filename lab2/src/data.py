import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class Instance:
    drivers: list[str]
    driver_zone: list[int]
    driver_start: list[float]
    driver_end: list[float]
    driver_rate: list[float]
    trips: list[str]
    trip_zone: list[int]
    trip_start: list[float]
    trip_end: list[float]
    trip_hours: np.ndarray
    arc_driver: list[int]
    arc_trip: list[int]
    arc_travel: np.ndarray
    arc_overlap: np.ndarray
    arc_capacity: np.ndarray
    arc_cost: np.ndarray

    @property
    def n_drivers(self) -> int:
        return len(self.drivers)

    @property
    def n_trips(self) -> int:
        return len(self.trips)

    @property
    def n_arcs(self) -> int:
        return len(self.arc_driver)

    @property
    def driver_hours(self) -> np.ndarray:
        return np.array(
            [end - start for start, end in zip(self.driver_start, self.driver_end)],
            dtype=float,
        )

    @property
    def total_demand(self) -> float:
        return float(np.sum(self.trip_hours))

    @property
    def total_supply(self) -> float:
        return float(np.sum(self.driver_hours))


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _fmt_hour(h: float) -> str:
    total = int(round(h * 60))
    return f"{total // 60:02d}:{total % 60:02d}"


def generate(seed: int, n_drivers: int = 6, n_trips: int = 6) -> Instance:
    rng = np.random.default_rng(seed)

    driver_templates = [
        (0, 6.0, 7.0),
        (1, 6.0, 7.0),
        (2, 8.0, 8.0),
        (3, 10.0, 8.0),
        (4, 12.0, 10.0),
        (2, 6.0, 16.0),
    ]
    trip_templates = [
        (0, 6.0, 6.0, 5.5),
        (1, 7.0, 7.0, 6.0),
        (2, 9.0, 6.0, 5.5),
        (3, 12.0, 6.0, 5.5),
        (4, 14.0, 6.0, 5.5),
        (1, 16.0, 6.0, 5.5),
    ]

    last_driver = driver_templates[-1]
    driver_templates = driver_templates[:n_drivers]
    while len(driver_templates) < n_drivers:
        driver_templates.append(last_driver)
    last_trip = trip_templates[-1]
    trip_templates = trip_templates[:n_trips]
    while len(trip_templates) < n_trips:
        trip_templates.append(last_trip)

    driver_zone = [t[0] for t in driver_templates]
    driver_start = [float(t[1]) for t in driver_templates]
    driver_end = [float(t[1] + t[2]) for t in driver_templates]
    driver_rate = [float(r) for r in np.round(rng.uniform(450.0, 720.0, n_drivers) / 10.0) * 10.0]

    trip_zone = [t[0] for t in trip_templates]
    trip_start = [float(t[1]) for t in trip_templates]
    trip_end = [float(t[1] + t[2]) for t in trip_templates]
    trip_hours = np.array([float(t[3]) for t in trip_templates], dtype=float)

    for j in range(n_trips):
        span = trip_end[j] - trip_start[j]
        if trip_hours[j] > span - 0.5:
            trip_hours[j] = round(span - 0.5, 1)

    arc_driver: list[int] = []
    arc_trip: list[int] = []
    arc_travel: list[float] = []
    arc_overlap: list[float] = []
    for i in range(n_drivers):
        for j in range(n_trips):
            travel = config.TRAVEL_H_PER_ZONE * abs(driver_zone[i] - trip_zone[j])
            arrive = driver_start[i] + travel
            dep = max(arrive, trip_start[j])
            leave = min(driver_end[i], trip_end[j])
            overlap = _overlap(dep, leave, trip_start[j], trip_end[j])
            if overlap >= trip_hours[j] - 1e-9 and driver_end[i] - driver_start[i] >= trip_hours[j] - 1e-9:
                arc_driver.append(i)
                arc_trip.append(j)
                arc_travel.append(travel)
                arc_overlap.append(overlap)

    arc_travel = np.array(arc_travel, dtype=float)
    arc_overlap = np.array(arc_overlap, dtype=float)
    arc_capacity = np.minimum(
        np.array([driver_end[i] - driver_start[i] for i in arc_driver], dtype=float),
        arc_overlap,
    )
    arc_cost = np.array(
        [driver_rate[i] * (1.0 + arc_travel[k] / trip_hours[arc_trip[k]]) for k, i in enumerate(arc_driver)],
        dtype=float,
    )

    inst = Instance(
        drivers=[f"Водитель {chr(65 + i)}" for i in range(n_drivers)],
        driver_zone=driver_zone,
        driver_start=driver_start,
        driver_end=driver_end,
        driver_rate=driver_rate,
        trips=[f"Рейс {j + 1}" for j in range(n_trips)],
        trip_zone=trip_zone,
        trip_start=trip_start,
        trip_end=trip_end,
        trip_hours=trip_hours,
        arc_driver=arc_driver,
        arc_trip=arc_trip,
        arc_travel=arc_travel,
        arc_overlap=arc_overlap,
        arc_capacity=arc_capacity,
        arc_cost=arc_cost,
    )
    validate(inst)
    return inst


def validate(inst: Instance) -> None:
    problems: list[str] = []
    supply_ratio = inst.total_supply / inst.total_demand
    if not 1.25 <= supply_ratio <= 1.7:
        problems.append(f"отношение мощности к спросу {supply_ratio:.2f} вне [1.25, 1.7]")

    for j in range(inst.n_trips):
        cands = [k for k in range(inst.n_arcs) if inst.arc_trip[k] == j]
        if len(cands) < 2:
            problems.append(f"{inst.trips[j]}: менее 2 допустимых водителей")
        covered = float(np.sum(inst.arc_capacity[cands])) if cands else 0.0
        if covered < 1.3 * inst.trip_hours[j]:
            problems.append(
                f"{inst.trips[j]}: покрытие {covered:.1f} ч < 1.3 × спрос {inst.trip_hours[j]:.1f} ч"
            )

    for i in range(inst.n_drivers):
        cands = [k for k in range(inst.n_arcs) if inst.arc_driver[k] == i]
        if not cands:
            problems.append(f"{inst.drivers[i]}: нет допустимых рейсов")

    narrow = sum(
        1
        for j in range(inst.n_trips)
        if sum(1 for k in range(inst.n_arcs) if inst.arc_trip[k] == j) <= 3
    )
    if narrow < 1:
        problems.append("нет узких мест: у всех рейсов более 3 кандидатов")

    if problems:
        raise ValueError("данные не прошли валидацию:\n- " + "\n- ".join(problems))


def save(inst: Instance, seed: int, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "drivers.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["driver_id", "name", "zone", "window_start", "window_end", "rate_rub_per_h"])
        for i in range(inst.n_drivers):
            w.writerow([
                i,
                inst.drivers[i],
                inst.driver_zone[i],
                _fmt_hour(inst.driver_start[i]),
                _fmt_hour(inst.driver_end[i]),
                f"{inst.driver_rate[i]:.1f}",
            ])

    with (data_dir / "trips.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["trip_id", "name", "zone", "window_start", "window_end", "hours"])
        for j in range(inst.n_trips):
            w.writerow([
                j,
                inst.trips[j],
                inst.trip_zone[j],
                _fmt_hour(inst.trip_start[j]),
                _fmt_hour(inst.trip_end[j]),
                f"{inst.trip_hours[j]:.1f}",
            ])

    with (data_dir / "arcs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["driver_id", "trip_id", "travel_h", "overlap_h", "capacity_h", "cost_rub_per_h"])
        for k in range(inst.n_arcs):
            w.writerow([
                inst.arc_driver[k],
                inst.arc_trip[k],
                f"{inst.arc_travel[k]:.2f}",
                f"{inst.arc_overlap[k]:.2f}",
                f"{inst.arc_capacity[k]:.2f}",
                f"{inst.arc_cost[k]:.2f}",
            ])

    meta = {
        "N": config.N,
        "Q": config.Q,
        "Y": config.Y,
        "variant": 1 + ((17 * config.N + 7 * config.Q + config.Y) % 20),
        "seed": seed,
        "n_drivers": inst.n_drivers,
        "n_trips": inst.n_trips,
        "n_arcs": inst.n_arcs,
        "total_supply_h": inst.total_supply,
        "total_demand_h": inst.total_demand,
        "generation": (
            "детерминированно из seed numpy.random.default_rng(seed): ставки водителей — "
            "равномерные 450..720 руб./ч с шагом 10; зоны, окна водителей и рейсов, "
            "длительности рейсов — из фиксированных шаблонов варианта; подача travel = "
            f"{config.TRAVEL_H_PER_ZONE} ч на зону; дуга (водитель, рейс) существует, если "
            "перекрытие доступного окна после подачи не меньше длительности рейса; "
            "ёмкость дуги = min(длина окна водителя, перекрытие); "
            "стоимость дуги = ставка × (1 + подача / длительность)."
        ),
        "units": {
            "windows": "ч (сутки, 24-часовой формат)",
            "hours": "ч",
            "rate": "руб./ч",
            "cost_arc": "руб./ч работы на рейсе",
            "objective": "руб.",
            "travel": "ч",
        },
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load(data_dir: Path) -> tuple[Instance, dict]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    def hour(text: str) -> float:
        h, m = text.split(":")
        return int(h) + int(m) / 60.0

    drivers: list[str] = []
    driver_zone: list[int] = []
    driver_start: list[float] = []
    driver_end: list[float] = []
    driver_rate: list[float] = []
    with (data_dir / "drivers.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            drivers.append(row["name"])
            driver_zone.append(int(row["zone"]))
            driver_start.append(hour(row["window_start"]))
            driver_end.append(hour(row["window_end"]))
            driver_rate.append(float(row["rate_rub_per_h"]))

    trips: list[str] = []
    trip_zone: list[int] = []
    trip_start: list[float] = []
    trip_end: list[float] = []
    trip_hours: list[float] = []
    with (data_dir / "trips.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            trips.append(row["name"])
            trip_zone.append(int(row["zone"]))
            trip_start.append(hour(row["window_start"]))
            trip_end.append(hour(row["window_end"]))
            trip_hours.append(float(row["hours"]))

    arc_driver: list[int] = []
    arc_trip: list[int] = []
    arc_travel: list[float] = []
    arc_overlap: list[float] = []
    arc_capacity: list[float] = []
    arc_cost: list[float] = []
    with (data_dir / "arcs.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            arc_driver.append(int(row["driver_id"]))
            arc_trip.append(int(row["trip_id"]))
            arc_travel.append(float(row["travel_h"]))
            arc_overlap.append(float(row["overlap_h"]))
            arc_capacity.append(float(row["capacity_h"]))
            arc_cost.append(float(row["cost_rub_per_h"]))

    inst = Instance(
        drivers=drivers,
        driver_zone=driver_zone,
        driver_start=driver_start,
        driver_end=driver_end,
        driver_rate=driver_rate,
        trips=trips,
        trip_zone=trip_zone,
        trip_start=trip_start,
        trip_end=trip_end,
        trip_hours=np.array(trip_hours, dtype=float),
        arc_driver=arc_driver,
        arc_trip=arc_trip,
        arc_travel=np.array(arc_travel, dtype=float),
        arc_overlap=np.array(arc_overlap, dtype=float),
        arc_capacity=np.array(arc_capacity, dtype=float),
        arc_cost=np.array(arc_cost, dtype=float),
    )
    return inst, meta


def ensure_data(seed: int, data_dir: Path, regenerate: bool = False) -> tuple[Instance, dict]:
    if regenerate or not (data_dir / "meta.json").exists():
        inst = generate(seed)
        save(inst, seed, data_dir)
    return load(data_dir)


def with_hours(inst: Instance, hours: np.ndarray) -> Instance:
    return Instance(
        drivers=inst.drivers,
        driver_zone=inst.driver_zone,
        driver_start=inst.driver_start,
        driver_end=inst.driver_end,
        driver_rate=inst.driver_rate,
        trips=inst.trips,
        trip_zone=inst.trip_zone,
        trip_start=inst.trip_start,
        trip_end=inst.trip_end,
        trip_hours=np.array(hours, dtype=float),
        arc_driver=inst.arc_driver,
        arc_trip=inst.arc_trip,
        arc_travel=inst.arc_travel,
        arc_overlap=inst.arc_overlap,
        arc_capacity=inst.arc_capacity,
        arc_cost=inst.arc_cost,
    )


def without_arcs(inst: Instance, drop: set[int]) -> Instance:
    keep = [k for k in range(inst.n_arcs) if k not in drop]
    return Instance(
        drivers=inst.drivers,
        driver_zone=inst.driver_zone,
        driver_start=inst.driver_start,
        driver_end=inst.driver_end,
        driver_rate=inst.driver_rate,
        trips=inst.trips,
        trip_zone=inst.trip_zone,
        trip_start=inst.trip_start,
        trip_end=inst.trip_end,
        trip_hours=inst.trip_hours.copy(),
        arc_driver=[inst.arc_driver[k] for k in keep],
        arc_trip=[inst.arc_trip[k] for k in keep],
        arc_travel=inst.arc_travel[keep].copy(),
        arc_overlap=inst.arc_overlap[keep].copy(),
        arc_capacity=inst.arc_capacity[keep].copy(),
        arc_cost=inst.arc_cost[keep].copy(),
    )


def with_arc_capacity(inst: Instance, overrides: dict[int, float]) -> Instance:
    cap = inst.arc_capacity.copy()
    for k, v in overrides.items():
        cap[k] = v
    return Instance(
        drivers=inst.drivers,
        driver_zone=inst.driver_zone,
        driver_start=inst.driver_start,
        driver_end=inst.driver_end,
        driver_rate=inst.driver_rate,
        trips=inst.trips,
        trip_zone=inst.trip_zone,
        trip_start=inst.trip_start,
        trip_end=inst.trip_end,
        trip_hours=inst.trip_hours.copy(),
        arc_driver=inst.arc_driver,
        arc_trip=inst.arc_trip,
        arc_travel=inst.arc_travel.copy(),
        arc_overlap=inst.arc_overlap.copy(),
        arc_capacity=cap,
        arc_cost=inst.arc_cost.copy(),
    )
