import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import platform
import scipy

from .data import Instance
from .model import driver_flows, trip_flows

SCENARIO_COLORS = {
    "base": "#4C72B0",
    "overload": "#DD8452",
    "demand_up": "#C44E52",
    "arc_failure": "#55A868",
    "stress": "#8172B3",
    "critical": "#CCB974",
}


def _num(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_solution_csv(inst: Instance, ver: dict, path: Path) -> None:
    lines = ["driver,trip,flow_h,capacity_h,utilization,cost_rub_per_h,saturated"]
    if ver.get("x") is None:
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    x = ver["x"]
    for k in range(inst.n_arcs):
        util = float(x[k]) / float(inst.arc_capacity[k]) if inst.arc_capacity[k] > 0 else 0.0
        saturated = int(x[k] >= inst.arc_capacity[k] - 1e-7)
        lines.append(
            f"{inst.drivers[inst.arc_driver[k]]},{inst.trips[inst.arc_trip[k]]},"
            f"{x[k]:.6f},{inst.arc_capacity[k]:.6f},{util:.4f},{inst.arc_cost[k]:.4f},{saturated}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verification_txt(ver: dict, path: Path) -> None:
    lines = [
        "Независимая проверка решения (NumPy, допуск 1e-7).",
        f"Статус решения: {ver['result'].status_text}",
        f"Objective из солвера: {ver['objective_solver']}",
        f"Objective пересчитан из x: {ver['objective_numpy']}",
        f"Разница objectives: {ver['objective_diff']}",
        "Проверки:",
    ]
    for c in ver["checks"]:
        lines.append(f"- {c['name']}: {'PASS' if c['ok'] else 'FAIL'}, max violation {c['violation']:.3e}")
    if not ver["checks"]:
        lines.append("- проверки не выполнялись: решение недоступно")
    lines.append(f"Независимая проверка допустимости: {'PASS' if ver['feasible'] else 'FAIL'}")
    if ver.get("driver_flow") is not None and ver.get("instance") is not None:
        inst = ver["instance"]
        lines.append("")
        lines.append("Баланс вершин-водителей (приток от источника, ч):")
        for i in range(inst.n_drivers):
            used = float(ver["driver_flow"][i])
            lines.append(
                f"- {inst.drivers[i]}: {used:.3f} / мощность {inst.driver_hours[i]:.3f} "
                f"(запас {inst.driver_hours[i] - used:.3f})"
            )
        lines.append("")
        lines.append("Баланс вершин-рейсов (приток = спрос, ч):")
        for j in range(inst.n_trips):
            got = float(ver["trip_flow"][j])
            lines.append(f"- {inst.trips[j]}: {got:.3f} / спрос {inst.trip_hours[j]:.3f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scenario_csv(results: list[dict], base_fun: float | None, path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["scenario", "label", "status", "objective", "delta", "delta_pct", "feasible", "description", "diagnostics"])
        for r in results:
            fun = r["result"].fun
            if fun is None or base_fun is None:
                delta, pct = "", ""
            else:
                delta = f"{fun - base_fun:.6f}"
                pct = f"{(fun - base_fun) / base_fun * 100.0:.4f}" if abs(base_fun) > 1e-12 else ""
            writer.writerow([
                r["name"],
                r["label"],
                r["result"].status_text,
                "" if fun is None else f"{fun:.6f}",
                delta,
                pct,
                r["verification"]["feasible"],
                r["description"],
                " | ".join(r.get("diagnostics", [])),
            ])


def write_shadows_csv(shadows: list[dict], path: Path) -> None:
    lines = ["label,type,rhs,marginal,shadow_price,active,slack"]
    for s in shadows:
        lines.append(
            f"{s['label']},{s['type']},{s['constraint_rhs']:.6f},{s['marginal']:.6f},"
            f"{s['shadow_price']:.6f},{int(s['active'])},{s['slack']:.6f}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_demand_duals_csv(rows: list[dict], path: Path) -> None:
    lines = ["label,rhs,dual_rub_per_h"]
    for r in rows:
        lines.append(f"{r['label']},{r['rhs']:.6f},{r['dual']:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_greedy_csv(rows: list[dict], path: Path) -> None:
    lines = ["scenario,lp_status,lp_cost,greedy_cost,greedy_unmet_h,gap,gap_pct,greedy_saturated_arcs"]
    for r in rows:
        gap = "" if r["lp_cost"] is None or r["greedy_cost"] is None else f"{r['greedy_cost'] - r['lp_cost']:.6f}"
        gap_pct = (
            ""
            if r["lp_cost"] is None or r["greedy_cost"] is None or abs(r["lp_cost"]) < 1e-12
            else f"{(r['greedy_cost'] - r['lp_cost']) / r['lp_cost'] * 100.0:.4f}"
        )
        lines.append(
            f"{r['scenario']},{r['lp_status']},"
            f"{'' if r['lp_cost'] is None else f'{r['lp_cost']:.6f}'},"
            f"{'' if r['greedy_cost'] is None else f'{r['greedy_cost']:.6f}'},"
            f"{r['greedy_unmet']:.6f},{gap},{gap_pct},{r['greedy_saturated']}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_milp_csv(row: dict, path: Path) -> None:
    lines = [
        "formulation,objective_rub,time_s,status,note",
        f"LP (дробные рейсы),{row['lp_fun']:.6f},{row['lp_time']:.6f},optimal,{row['lp_note']}",
        f"MILP (неделимые рейсы),{row['milp_fun']:.6f},{row['milp_time']:.6f},{row['milp_status']},{row['milp_note']}",
        f"gap (MILP-LP),{row['gap_abs']:.6f},,,{row['gap_pct']:.4f}% стоимости",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dual_check_txt(dc: dict, path: Path) -> None:
    if not dc.get("available"):
        Path(path).write_text("Двойственная проверка недоступна.\n", encoding="utf-8")
        return
    lines = [
        "Проверка двойственности (сильная двойственность и дополняющая двойственность).",
        f"Primal objective: {dc['primal']:.10f}",
        f"Dual objective:   {dc['dual']:.10f}",
        f"|primal - dual|:  {dc['gap']:.3e}",
        f"Max complementary slackness violation: {dc['max_cs_violation']:.3e}",
        f"Max reduced cost violation (rc < 0):    {dc['max_rc_violation']:.3e}",
        "",
        "Дуга, поток, rc, запас по мощности водителя, запас по ёмкости дуги:",
    ]
    for r in dc["rows"]:
        lines.append(
            f"- {r['arc']}: x={r['flow']:.3f}, rc={r['reduced_cost']:.6f}, "
            f"slack_driver={r['slack_driver']:.3f}, slack_arc={r['slack_arc']:.3f}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _align_scenario_arcs(
    base: Instance, scen: Instance, x_scen: np.ndarray
) -> tuple[np.ndarray, set[int]]:
    """Возвращает потоки сценария в индексах базовой дуги и множество закрытых дуг."""
    x_full = np.zeros(base.n_arcs)
    base_key_to_idx = {(base.arc_driver[k], base.arc_trip[k]): k for k in range(base.n_arcs)}
    matched: set[int] = set()
    for k2 in range(scen.n_arcs):
        k1 = base_key_to_idx[(scen.arc_driver[k2], scen.arc_trip[k2])]
        matched.add(k1)
        x_full[k1] = x_scen[k2]
    closed = set(range(base.n_arcs)) - matched
    return x_full, closed


def _network_positions(inst: Instance) -> dict[str, tuple[float, float]]:
    pos: dict[str, tuple[float, float]] = {"S": (0.0, 0.0), "K": (3.0, 0.0)}
    nd, nt = inst.n_drivers, inst.n_trips
    for i in range(nd):
        pos[f"D{i}"] = (1.0, (nd - 1) / 2.0 - i)
    for j in range(nt):
        pos[f"T{j}"] = (2.0, (nt - 1) / 2.0 - j)
    return pos


def draw_network(
    inst: Instance,
    x: np.ndarray,
    path: Path,
    title: str,
    closed_arcs: set[int] | None = None,
) -> None:
    closed_arcs = closed_arcs or set()
    G = nx.DiGraph()
    driver_flow = driver_flows(inst, x)
    trip_flow = trip_flows(inst, x)

    for i in range(inst.n_drivers):
        G.add_edge("S", f"D{i}", kind="sd", flow=float(driver_flow[i]), cap=float(inst.driver_hours[i]))
    for k in range(inst.n_arcs):
        G.add_edge(
            f"D{inst.arc_driver[k]}",
            f"T{inst.arc_trip[k]}",
            kind="dt",
            flow=float(x[k]),
            cap=float(inst.arc_capacity[k]),
            cost=float(inst.arc_cost[k]),
            arc=k,
            closed=k in closed_arcs,
        )
    for j in range(inst.n_trips):
        G.add_edge(f"T{j}", "K", kind="tk", flow=float(inst.trip_hours[j]), cap=float(inst.trip_hours[j]))

    pos = _network_positions(inst)
    fig, ax = plt.subplots(figsize=(13.5, 8.5))

    node_colors = []
    node_labels = {}
    for n in G.nodes:
        if n == "S":
            node_colors.append("#555555")
            node_labels[n] = "S\nисточник\n(часы смен)"
        elif n == "K":
            node_colors.append("#555555")
            node_labels[n] = "K\nсток\n(рейсы)"
        elif n.startswith("D"):
            i = int(n[1:])
            node_colors.append("#4C72B0")
            node_labels[n] = f"{inst.drivers[i]}\n{inst.driver_rate[i]:.0f} ₽/ч\nокно {inst.driver_hours[i]:.0f} ч"
        else:
            j = int(n[1:])
            node_colors.append("#DD8452")
            node_labels[n] = f"{inst.trips[j]}\n{inst.trip_hours[j]:.1f} ч"

    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        if data["kind"] == "dt":
            if data["closed"]:
                edge_colors.append("#000000")
                edge_widths.append(1.0)
            elif data["flow"] <= 1e-7:
                edge_colors.append("#D0D0D0")
                edge_widths.append(0.8)
            elif data["flow"] >= data["cap"] - 1e-7:
                edge_colors.append("#C44E52")
                edge_widths.append(1.0 + 3.5 * data["flow"] / max(inst.trip_hours.max(), 1.0))
            else:
                edge_colors.append("#2b6cb0")
                edge_widths.append(1.0 + 3.5 * data["flow"] / max(inst.trip_hours.max(), 1.0))
        else:
            edge_colors.append("#999999")
            edge_widths.append(1.0 + 2.5 * data["flow"] / max(inst.trip_hours.max(), 1.0))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2600, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_color="white", ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=edge_widths, arrows=True,
        arrowsize=14, connectionstyle="arc3,rad=0.06", ax=ax, min_source_margin=22, min_target_margin=22,
    )

    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if data["kind"] == "sd":
            edge_labels[(u, v)] = f"{data['flow']:.1f}/{data['cap']:.0f} ч"
        elif data["kind"] == "dt":
            if data["closed"]:
                edge_labels[(u, v)] = "закрыта"
            elif data["flow"] > 1e-7:
                edge_labels[(u, v)] = f"{data['flow']:.1f}/{data['cap']:.1f} ч\n{data['cost']:.0f} ₽/ч"
            else:
                edge_labels[(u, v)] = f"0/{data['cap']:.1f}"
        else:
            edge_labels[(u, v)] = f"{data['flow']:.1f} ч"

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=6, ax=ax, rotate=False,
        label_pos=0.42, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
    )

    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(
    inst: Instance,
    base_ver: dict,
    scenario_results: list[dict],
    shadows: list[dict],
    fig_dir: Path,
    greedy_rows: list[dict],
    milp_row: dict,
    base_x: np.ndarray,
) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    base_path = fig_dir / "network_base.png"
    draw_network(inst, base_x, base_path, "Базовая сеть: потоки (ч) / ёмкости (ч), стоимость дуг — ₽/ч")
    made.append(base_path.name)

    arc_fail = next((r for r in scenario_results if r["name"] == "arc_failure"), None)
    if arc_fail is not None and arc_fail["verification"]["x"] is not None:
        x_full, closed = _align_scenario_arcs(inst, arc_fail["instance"], arc_fail["verification"]["x"])
        path = fig_dir / "network_arc_failure.png"
        draw_network(
            inst,
            x_full,
            path,
            "Сеть после отказа самой загруженной дуги: перераспределение потоков",
            closed_arcs=closed,
        )
        made.append(path.name)

    names = [r["label"] for r in scenario_results]
    funs = [r["result"].fun if r["result"].fun is not None else np.nan for r in scenario_results]
    colors = [SCENARIO_COLORS.get(r["name"], "#888888") for r in scenario_results]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(np.arange(len(names)), np.nan_to_num(funs, nan=0.0), color=colors)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Целевая функция, руб.")
    ax.set_title("Стоимость по сценариям")
    for i, r in enumerate(scenario_results):
        if r["result"].fun is None:
            ax.text(i, 0, "infeasible", ha="center", va="bottom", fontsize=9, color="#C44E52")
        else:
            ax.text(i, funs[i], f"{funs[i]:.0f}", ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.12)
    fig.tight_layout()
    out = fig_dir / "scenarios.png"
    fig.savefig(out)
    plt.close(fig)
    made.append(out.name)

    active = [s for s in shadows if s["active"]]
    if active:
        rows = sorted(active, key=lambda s: s["shadow_price"])
        labels = [
            (s["label"].replace("мощность[", "окно ").replace("дуга[", "дуга ").rstrip("]"))
            for s in rows
        ]
        vals = [s["shadow_price"] for s in rows]
        colors_b = ["#4C72B0" if s["type"] == "driver_capacity" else "#C44E52" for s in rows]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bars = ax.barh(labels, vals, color=colors_b)
        ax.set_xlabel("Теневая цена, руб. за +1 ч ёмкости")
        ax.set_title("Теневые цены активных ограничений (базовый сценарий)")
        for bar, value in zip(bars, vals):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center", fontsize=8)
        ax.margins(x=0.18)
        fig.tight_layout()
        out = fig_dir / "bottlenecks.png"
        fig.savefig(out)
        plt.close(fig)
        made.append(out.name)

    if arc_fail is not None and arc_fail["verification"]["x"] is not None:
        x_fail_full, _closed_fail = _align_scenario_arcs(
            inst, arc_fail["instance"], arc_fail["verification"]["x"]
        )
        changed = [k for k in range(inst.n_arcs) if abs(float(x_fail_full[k]) - float(base_x[k])) > 1e-6]
        if changed:
            labels = [
                f"{inst.drivers[inst.arc_driver[k]].replace('Водитель ', '')}→{inst.trips[inst.arc_trip[k]].replace('Рейс ', 'R')}"
                for k in changed
            ]
            xs = np.arange(len(changed))
            width = 0.38
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.bar(xs - width / 2, [base_x[k] for k in changed], width, label="базовый", color="#4C72B0")
            ax.bar(xs + width / 2, [x_fail_full[k] for k in changed], width, label="отказ дуги", color="#C44E52")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=0, fontsize=8)
            ax.set_ylabel("Поток, ч")
            ax.set_title("Перераспределение потоков после отказа дуги")
            ax.legend()
            fig.tight_layout()
            out = fig_dir / "arc_failure_flows.png"
            fig.savefig(out)
            plt.close(fig)
            made.append(out.name)

    if milp_row is not None:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        labels = ["LP (дробные рейсы)", "Жадная эвристика", "MILP (неделимые)"]
        vals = [milp_row["lp_fun"], milp_row["greedy_fun"], milp_row["milp_fun"]]
        cols = ["#4C72B0", "#DD8452", "#55A868"]
        ax.bar(labels, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Целевая функция, руб.")
        ax.set_title("LP vs жадная эвристика vs MILP (базовый сценарий)")
        ax.margins(y=0.12)
        fig.tight_layout()
        out = fig_dir / "methods.png"
        fig.savefig(out)
        plt.close(fig)
        made.append(out.name)

    return made


def build_report(
    inst: Instance,
    meta: dict,
    base_ver: dict,
    dual: dict,
    scenario_results: list[dict],
    shadows: list[dict],
    demand_duals: list[dict],
    greedy_rows: list[dict],
    milp_row: dict,
    figures: list[str],
    fig_prefix: str,
) -> str:
    base_fun = base_ver["objective_solver"]
    x = base_ver["x"]
    driver_flow = base_ver["driver_flow"]

    def scenario(name: str) -> dict | None:
        return next((r for r in scenario_results if r["name"] == name), None)

    lines: list[str] = []
    lines.append("# Отчёт по лабораторной работе № 2. Транспортные и сетевые модели")
    lines.append("")
    lines.append("## 1. Постановка задачи и индивидуальный вариант")
    lines.append("")
    lines.append(
        f"Число варианта: V = {meta['variant']}; студент N = {meta['N']}; группа Q = {meta['Q']}; "
        f"год Y = {meta['Y']}; seed = {meta['seed']}."
    )
    lines.append("")
    lines.append(
        f"Задача варианта 7: назначение водителей на рейсы с окнами времени "
        f"({inst.n_drivers} водителей, {inst.n_trips} рейсов). Требуется минимизировать совокупные "
        "эксплуатационные затраты, распределив часы работы водителей по рейсам так, чтобы каждый рейс "
        "был полностью покрыт, мощность каждого водителя (длина окна) не была превышена, а назначение "
        "было допустимо по пересечению окон времени с учётом подачи транспорта."
    )
    lines.append("")
    lines.append(
        "Сеть: исток S → вершины водителей → вершины рейсов → сток K; дуги S→водитель имеют ёмкость "
        "равную длине окна водителя, дуги водитель→рейс существуют только при совместимости окон и "
        "имеют ёмкость и стоимость, дуги рейс→K несут спрос рейса."
    )
    lines.append("")

    lines.append("## 2. Данные варианта и процедура генерации")
    lines.append("")
    lines.append(meta["generation"])
    lines.append("")
    driver_rows = [
        [
            inst.drivers[i],
            str(inst.driver_zone[i]),
            f"{inst.driver_start[i]:02.0f}:00–{inst.driver_end[i]:02.0f}:00",
            f"{inst.driver_hours[i]:.0f}",
            f"{inst.driver_rate[i]:.0f}",
        ]
        for i in range(inst.n_drivers)
    ]
    lines.append(_md_table(["Водитель", "Зона", "Окно, ч", "Мощность, ч", "Ставка, ₽/ч"], driver_rows))
    lines.append("")
    trip_rows = [
        [
            inst.trips[j],
            str(inst.trip_zone[j]),
            f"{inst.trip_start[j]:02.0f}:00–{inst.trip_end[j]:02.0f}:00",
            f"{inst.trip_hours[j]:.1f}",
            str(sum(1 for k in range(inst.n_arcs) if inst.arc_trip[k] == j)),
        ]
        for j in range(inst.n_trips)
    ]
    lines.append(_md_table(["Рейс", "Зона", "Окно, ч", "Спрос, ч", "Допустимых водителей"], trip_rows))
    lines.append("")
    lines.append(
        f"Всего допустимых дуг: {inst.n_arcs} из {inst.n_drivers * inst.n_trips} возможных пар "
        f"(совместимость окон отсекает несовместимые назначения). "
        f"Суммарная мощность водителей: {inst.total_supply:.0f} ч; суммарный спрос: {inst.total_demand:.1f} ч "
        f"(отношение {inst.total_supply / inst.total_demand:.2f})."
    )
    lines.append("")
    lines.append(_md_table(["Параметр", "Единица измерения"], [
        ["H_i — мощность водителя (длина окна)", "ч"],
        ["h_j — спрос рейса (длительность)", "ч"],
        ["u_ij — ёмкость дуги (пересечение окон с учётом подачи)", "ч"],
        ["t_ij — подача (время переезда в зону рейса)", "ч"],
        ["c_ij — стоимость часа работы водителя i на рейсе j", "руб./ч"],
        ["rate_i — почасовая ставка водителя", "руб./ч"],
        ["x_ij — поток по дуге (часы на рейсе)", "ч"],
        ["Целевая функция", "руб."],
    ]))
    lines.append("")

    lines.append("## 3. Формальная модель")
    lines.append("")
    lines.append(
        f"Множества: водители D = {{{', '.join(inst.drivers)}}}; рейсы T = {{{', '.join(inst.trips)}}}; "
        f"допустимые дуги A ⊆ D × T (|A| = {inst.n_arcs})."
    )
    lines.append("")
    lines.append(
        "Переменные: x_ij ≥ 0 — часы работы водителя i на рейсе j, ч (непрерывная переменная потока)."
    )
    lines.append("")
    lines.append(
        "Параметры: H_i — мощность окна водителя, ч; h_j — длительность рейса, ч; "
        "u_ij — ёмкость дуги, ч; c_ij = rate_i · (1 + t_ij / h_j) — стоимость часа потока, руб./ч."
    )
    lines.append("")
    lines.append("min Σ_(i,j)∈A c_ij · x_ij")
    lines.append("")
    lines.append("- Баланс рейса (спрос): для каждого рейса j ∈ T суммарный поток равен длительности: Σ_i x_ij = h_j.")
    lines.append("- Мощность водителя: для каждого водителя i ∈ D суммарные часы не превышают окно: Σ_j x_ij ≤ H_i.")
    lines.append("- Границы дуг: 0 ≤ x_ij ≤ u_ij (работа в пределах совместимого пересечения окон).")
    lines.append("- Неотрицательность: x_ij ≥ 0.")
    lines.append("")
    lines.append(
        "Ограничения баланса — типичные узловые условия потока; вместе с источником S (дуги S→i с ёмкостью H_i) "
        "и стоком K (дуги j→K с потоком h_j) модель является задачей о минимальной стоимости потока в сети."
    )
    lines.append("")

    lines.append("## 4. Двойственная задача и интерпретация")
    lines.append("")
    lines.append(
        "Двойственная задача: max Σ_j β_j·h_j − Σ_i α_i·H_i − Σ_(i,j) γ_ij·u_ij при ограничениях "
        "β_j − α_i − γ_ij ≤ c_ij для всех дуг (i,j) ∈ A, α ≥ 0, γ ≥ 0, β свободны по знаку."
    )
    lines.append("")
    lines.append(
        "Смысл: β_j — потенциал рейса j, предельная (предельно допустимая) стоимость часа спроса рейса; "
        "α_i — дефицит (теневая цена) часа окна водителя i, на сколько снизились бы затраты при +1 ч мощности; "
        "γ_ij — ценность ёмкости дуги. Для несвязанных (нулевых) потоков выполняется "
        "сокращённая стоимость rc_ij = c_ij − β_j + α_i + γ_ij ≥ 0."
    )
    lines.append("")
    if dual.get("available"):
        lines.append(
            f"Численная проверка: primal = {dual['primal']:.6f} руб., dual = {dual['dual']:.6f} руб., "
            f"|primal − dual| = {dual['gap']:.2e}; максимальное нарушение дополняющей двойственности "
            f"{dual['max_cs_violation']:.2e}, нарушений rc < 0: {dual['max_rc_violation']:.2e}."
        )
        lines.append("")
        active_arcs = [r for r in dual["rows"] if r["flow"] > 1e-7]
        idle_arcs = [r for r in dual["rows"] if r["flow"] <= 1e-7]
        lines.append(
            f"Дуги с ненулевым потоком ({len(active_arcs)}) имеют rc = 0 (в пределах точности); "
            f"дуги без потока ({len(idle_arcs)}) имеют rc ≥ 0 — их использование неоптимально."
        )
        lines.append("")
    lines.append(_md_table(
        ["Потенциал рейса β_j", "Значение, ₽/ч"],
        [[r["label"].replace("спрос[", "").rstrip("]"), f"{r['dual']:.2f}"] for r in demand_duals],
    ))
    lines.append("")
    if demand_duals:
        best_d = max(demand_duals, key=lambda r: r["dual"])
        worst_d = min(demand_duals, key=lambda r: r["dual"])
        clean = lambda label: label.replace("спрос[", "").rstrip("]")
        lines.append(
            "Потенциалы показывают предельную стоимость обслуживания каждого рейса: самый дорогой — "
            f"{clean(best_d['label'])} ({best_d['dual']:.1f} ₽/ч), самый дешёвый — "
            f"{clean(worst_d['label'])} ({worst_d['dual']:.1f} ₽/ч)."
        )
        lines.append("")

    lines.append("## 5. Допущения и учебные упрощения")
    lines.append("")
    lines.append(
        "1) ЛП допускает дробное разделение рейса между водителями (x_ij — часы); на практике рейс неделим, "
        "что проверяется в дополнительном задании (MILP). "
        "2) Затраты линейны: стоимость часа потока c_ij включает ставку водителя и подачу, отнесённую на часы рейса; "
        "подача при дроблении делится пропорционально часам. "
        "3) Окна времени жёсткие: дуга существует только если водитель успевает выполнить рейс целиком "
        "(пересечение доступного окна после подачи ≥ длительность рейса). "
        "4) Модель не учитывает точный график внутри дня при нескольких рейсах одного водителя: "
        "ограничение — сумма часов ≤ длина окна; последовательность и отсутствие наложений "
        "предполагаются согласованными диспетчером. "
        "5) Нет неопределённости спроса, задержек и простоев; затраты детерминированы."
    )
    lines.append("")

    lines.append("## 6. Реализация")
    lines.append("")
    lines.append(
        "Модель собирается из матриц A_eq/A_ub и решается SciPy `linprog` с методом HiGHS "
        "(`method='highs-ds'`, dual simplex), options: presolve=True, disp=False. "
        f"Версии: Python {platform.python_version()}, SciPy {scipy.__version__}, "
        f"NetworkX {nx.__version__}; HiGHS поставляется в составе SciPy, версии зафиксированы в requirements.txt."
    )
    lines.append("")
    lines.append(
        "Интерфейс: `build_model(Instance) → LPModel(c, A_ub, b_ub, A_eq, b_eq, bounds, "
        "constraint_labels, equality_labels, arc_capacity_rows, driver_rows)`; "
        "переменные индексируются дугами сети."
    )
    lines.append("")
    lines.append(
        f"Базовый сценарий: статус {scenario_results[0]['result'].status_text}, "
        f"целевая функция {_num(base_fun)} руб., итераций {scenario_results[0]['result'].nit}, "
        f"время решения {scenario_results[0]['result'].solve_time_s * 1000:.2f} мс."
    )
    lines.append("")

    lines.append("## 7. Статус решения и интерпретация")
    lines.append("")
    lines.append(f"Статус: {scenario_results[0]['result'].status_text}. Целевая функция: {_num(base_fun)} руб.")
    lines.append("")
    lines.append(
        "Статус `optimal` означает найденное допустимое решение, минимизирующее затраты: все узловые балансы "
        "и границы дуг выполнены с допуском 1e-7. Статус `infeasible` означает несовместимость ограничений "
        "(спрос не покрывается мощностью и совместимыми окнами) и требует диагностики — она выполнена "
        "в критическом сценарии. Статус `unbounded` для этой модели невозможен: целевая функция ограничена снизу "
        "нулёми, а сверху — конечной мощностью сети."
    )
    lines.append("")

    lines.append("## 8. Независимая проверка")
    lines.append("")
    lines.append(_md_table(
        ["check", "ok", "violation"],
        [[c["name"], str(c["ok"]), f"{c['violation']:.3e}"] for c in base_ver["checks"]],
    ))
    lines.append("")
    lines.append(
        f"Objective из солвера: {_num(base_ver['objective_solver'])}; пересчитан из x средствами NumPy: "
        f"{_num(base_ver['objective_numpy'])}; разница: {base_ver['objective_diff']:.3e}. "
        "Проверки выполняются отдельным кодом без обращения к объекту солвера: баланс каждой вершины "
        "(водитель: приток от источника = сумма потоков и ≤ мощности; рейс: приток = отток = спрос; "
        "исток и сток: отток = приток), границы каждой дуги (0 ≤ x ≤ u) и пересчёт цели."
    )
    lines.append("")
    flow_rows = []
    for i in range(inst.n_drivers):
        used = float(driver_flow[i])
        flow_rows.append([
            inst.drivers[i],
            f"{used:.2f}",
            f"{inst.driver_hours[i]:.1f}",
            f"{inst.driver_hours[i] - used:.2f}",
            "да" if used >= inst.driver_hours[i] - 1e-7 else "нет",
        ])
    lines.append(_md_table(["Водитель", "Загрузка, ч", "Мощность, ч", "Запас, ч", "Ограничение активно"], flow_rows))
    lines.append("")

    lines.append("## 9. Схема сети и найденные потоки")
    lines.append("")
    if "network_base.png" in figures:
        lines.append(f"![network_base.png]({fig_prefix}network_base.png)")
        lines.append("")
    lines.append(
        "На схеме: толщина дуги пропорциональна потоку, красным отмечены насыщенные дуги (поток = ёмкость), "
        "серым — допустимые дуги без потока; подписи — поток/ёмкость в часах, на дугах водитель→рейс также "
        "указана стоимость, ₽/ч."
    )
    lines.append("")
    sol_rows = []
    for k in range(inst.n_arcs):
        util = x[k] / inst.arc_capacity[k] if inst.arc_capacity[k] > 0 else 0.0
        sol_rows.append([
            f"{inst.drivers[inst.arc_driver[k]]} → {inst.trips[inst.arc_trip[k]]}",
            f"{x[k]:.2f}",
            f"{inst.arc_capacity[k]:.2f}",
            f"{util * 100:.0f}%",
            f"{inst.arc_cost[k]:.1f}",
            f"{x[k] * inst.arc_cost[k]:.2f}",
        ])
    lines.append(_md_table(["Дуга", "Поток, ч", "Ёмкость, ч", "Загрузка", "₽/ч", "Затраты, ₽"], sol_rows))
    lines.append("")

    lines.append("## 10. Сценарный анализ")
    lines.append("")
    scen_rows = []
    for r in scenario_results:
        fun = r["result"].fun
        if fun is None or base_fun is None:
            delta, pct = "—", "—"
        else:
            delta = f"{fun - base_fun:+.2f}"
            pct = f"{(fun - base_fun) / base_fun * 100:+.2f}%"
        scen_rows.append([
            r["label"],
            r["result"].status_text,
            _num(fun),
            delta,
            pct,
            str(r["verification"]["feasible"]),
        ])
    lines.append(_md_table(["Сценарий", "Статус", "Цель, ₽", "Δ", "Δ%", "Проверка"], scen_rows))
    lines.append("")
    for r in scenario_results:
        lines.append(f"- **{r['label']}**: {r['description']}")
    lines.append("")

    overload = scenario("overload")
    demand_up = scenario("demand_up")
    arc_fail = scenario("arc_failure")
    stress = scenario("stress")
    critical = scenario("critical")

    if overload is not None and overload["result"].fun is not None:
        od = overload["result"].fun - base_fun
        lines.append(
            f"Перегрузка дуги (ёмкость снижена на 50%): ΔC = {od:+.2f} руб. "
            "Ущерб мал, поскольку лимитирующим ресурсом в базовом решении было окно водителя, "
            "а не ёмкость дуги: поток перераспределяется по другим дугам совместимых водителей, "
            "и запас ёмкости поглощает сокращение."
        )
        lines.append("")
    if demand_up is not None and demand_up["result"].fun is not None and dual.get("available"):
        dd = demand_up["result"].fun - base_fun
        beta = np.asarray(dual["beta"], dtype=float)
        delta_h = demand_up["instance"].trip_hours - inst.trip_hours
        pred_lower = float(beta @ delta_h)
        lines.append(
            f"Рост спроса на 20%: ΔC = {dd:+.2f} руб. ({dd / base_fun * 100:+.2f}%). "
            f"Прогноз по базовым потенциалам: βᵀΔh = {pred_lower:+.2f} руб. "
            "Двойственная точка базового решения остаётся допустимой при изменении правых частей, "
            f"поэтому прогноз — гарантированная нижняя граница роста затрат; факт ({dd:+.2f}) превышает её, "
            "что означает смену активного множества (подключились ранее неиспользуемые дуги и водители)."
        )
        lines.append("")
        du_active = [s for s in demand_up_shadows(demand_up) if s["active"] and s["type"] == "arc_capacity"]
        if du_active:
            lines.append(
                "В дефицитном сценарии дополнительно насыщаются дуги: "
                + ", ".join(s["label"].replace("дуга[", "").rstrip("]") for s in du_active)
                + " — новые узкие места сети."
            )
            lines.append("")
    if arc_fail is not None and arc_fail["result"].fun is not None:
        fd = arc_fail["result"].fun - base_fun
        fail_arc_label = arc_fail["description"].split("закрыта ")[-1].split(" (")[0]
        lines.append(
            f"Отказ дуги ({fail_arc_label}): ΔC = {fd:+.2f} руб. ({fd / base_fun * 100:+.2f}%) — "
            "прямой ущерб от потери самого загруженного назначения; рейс обслуживается альтернативным "
            "водителем с более высокой стоимостью. Теневая цена ёмкости закрытой дуги в базовом решении "
            "равна нулю (запас по ёмкости), поэтому локальный прогноз не улавливает ущерб: отказ дуги — "
            "грубое изменение структуры сети, выходящее из области действия теневой цены."
        )
        lines.append("")
        if "network_arc_failure.png" in figures:
            lines.append(f"![network_arc_failure.png]({fig_prefix}network_arc_failure.png)")
            lines.append("")
        if "arc_failure_flows.png" in figures:
            lines.append(f"![arc_failure_flows.png]({fig_prefix}arc_failure_flows.png)")
            lines.append("")
    if stress is not None and stress["result"].fun is not None:
        sd = stress["result"].fun - base_fun
        lines.append(
            f"Стрессовый сценарий (отказ дуги + спрос +20%): ΔC = {sd:+.2f} руб. "
            "Ограничения накладываются: отказ сужает структуру сети, рост спроса увеличивает требования — "
            "совокупный эффект чуть больше суммы отдельных эффектов."
        )
        lines.append("")
    if critical is not None and critical["result"].status != 0:
        lines.append(
            f"Критический сценарий (спрос +50% и закрыты две самые загруженные дуги): "
            f"статус {critical['result'].status_text}. Диагностика несовместимости:"
        )
        for issue in critical["diagnostics"][:6]:
            lines.append(f"- {issue}")
        if len(critical["diagnostics"]) > 6:
            lines.append(f"- … и ещё {len(critical['diagnostics']) - 6} нарушений (полный список в scenarios.csv)")
        lines.append("")
        lines.append(
            "Вывод: ограничения конфликтуют не только по суммарной мощности, но и локально — "
            "закрытые дуги и окна времени оставляют часть рейсов без допустимых исполнителей; "
            "при infeasible солвер не даёт плана, а диагностика указывает конкретные конфликтующие подмножества."
        )
        lines.append("")
    if "scenarios.png" in figures:
        lines.append(f"![scenarios.png]({fig_prefix}scenarios.png)")
        lines.append("")

    lines.append("## 11. Узкие места сети и теневые цены")
    lines.append("")
    active = [s for s in shadows if s["active"]]
    if active:
        lines.append(_md_table(
            ["Ограничение", "rhs, ч", "Теневая цена, ₽/ч", "Запас, ч"],
            [
                [s["label"], f"{s['constraint_rhs']:.2f}", f"{s['shadow_price']:.2f}", f"{s['slack']:.3f}"]
                for s in sorted(active, key=lambda s: -s["shadow_price"])
            ],
        ))
        lines.append("")
        best = max(active, key=lambda s: s["shadow_price"])
        lines.append(
            f"Главное узкое место базового сценария: {best['label']} — +1 ч мощности снизил бы затраты "
            f"на {best['shadow_price']:.2f} руб./ч. Ненулевые теневые цены имеют только активные ограничения; "
            "остальные запасы избыточны."
        )
        lines.append("")
        sat_arcs = [
            k for k in range(inst.n_arcs) if x[k] >= inst.arc_capacity[k] - 1e-7
        ]
        near_arcs = sorted(
            [k for k in range(inst.n_arcs) if x[k] > 1e-7],
            key=lambda k: -(x[k] / inst.arc_capacity[k] if inst.arc_capacity[k] > 0 else 0),
        )[:3]
        if sat_arcs:
            lines.append(
                "Насыщенные дуги (поток = ёмкость): "
                + ", ".join(
                    f"{inst.drivers[inst.arc_driver[k]]}→{inst.trips[inst.arc_trip[k]]} "
                    f"({x[k]:.1f}/{inst.arc_capacity[k]:.1f} ч)"
                    for k in sat_arcs
                )
                + "."
            )
        else:
            lines.append(
                "В базовом решении нет дуг с потоком, равным ёмкости: лимитирующими оказываются окна "
                "водителей, а пересечения окон дают небольшой запас. Наиболее загруженные дуги: "
                + ", ".join(
                    f"{inst.drivers[inst.arc_driver[k]]}→{inst.trips[inst.arc_trip[k]]} "
                    f"({x[k]:.1f}/{inst.arc_capacity[k]:.1f} ч, {x[k] / inst.arc_capacity[k] * 100:.0f}%)"
                    for k in near_arcs
                )
                + "."
            )
        lines.append("")
        if "bottlenecks.png" in figures:
            lines.append(f"![bottlenecks.png]({fig_prefix}bottlenecks.png)")
            lines.append("")
        lines.append(
            "Связь с теневыми ценами: дуги с запасом ёмкости имеют γ_ij = 0, поэтому их локальное "
            "расширение бесполезно; ресурс, за который стоит бороться, — часы окон активных водителей "
            "(α_i > 0). В дефицитном сценарии (см. раздел 10) насыщаются и дуги — их γ становится положительным."
        )
        lines.append("")
    else:
        lines.append("Активных ограничений нет.")
        lines.append("")

    lines.append("## 12. Сравнение с жадной эвристикой")
    lines.append("")
    greedy_rows_sorted = greedy_rows
    label_by_name = {r["name"]: r["label"] for r in scenario_results}
    g_rows = [
        [
            label_by_name.get(r["scenario"], r["scenario"]),
            r["lp_status"],
            _num(r["lp_cost"]),
            _num(r["greedy_cost"]) if r["lp_cost"] is not None else "—",
            "—"
            if r["lp_cost"] is None or r["greedy_cost"] is None
            else f"{(r['greedy_cost'] - r['lp_cost']) / r['lp_cost'] * 100:.3f}%",
            "—" if r["lp_cost"] is None else f"{r['greedy_unmet']:.2f}",
        ]
        for r in greedy_rows_sorted
    ]
    lines.append(_md_table(
        ["Сценарий", "LP статус", "LP, ₽", "Жадный, ₽", "Зазор", "Не покрыто жадным, ч"],
        g_rows,
    ))
    lines.append("")
    base_g = next((r for r in greedy_rows if r["scenario"] == "base"), None)
    if base_g and base_g["lp_cost"] is not None and base_g["greedy_cost"] is not None:
        gap_pct = (base_g["greedy_cost"] - base_g["lp_cost"]) / base_g["lp_cost"] * 100
        lines.append(
            f"Жадная эвристика («сначала самые дешёвые дуги»: сортировка по c_ij, налив потока "
            f"min(остаток мощности, остаток спроса, ёмкость)) даёт стоимость {base_g['greedy_cost']:.2f} руб. "
            f"против оптимума {base_g['lp_cost']:.2f} руб.: зазор {gap_pct:.3f}%. "
            "Эвристика полностью покрывает спрос, но жадный порядок не учитывает дефицит дешёвой мощности "
            "на будущие (более дорогие) рейсы, поэтому уступает оптимуму."
        )
        lines.append("")

    lines.append("## 13. Содержательный вывод")
    lines.append("")
    lines.append(
        f"Оптимальные суммарные затраты базового сценария: {_num(base_fun)} руб. "
        f"({inst.n_trips} рейсов, {inst.total_demand:.1f} ч работы)."
    )
    lines.append("")
    assign_rows = []
    for j in range(inst.n_trips):
        parts = [
            f"{inst.drivers[inst.arc_driver[k]]}: {x[k]:.1f} ч"
            for k in range(inst.n_arcs)
            if inst.arc_trip[k] == j and x[k] > 1e-7
        ]
        assign_rows.append([inst.trips[j], f"{inst.trip_hours[j]:.1f}", "; ".join(parts)])
    lines.append(_md_table(["Рейс", "Спрос, ч", "Назначение"], assign_rows))
    lines.append("")
    idle = [
        inst.drivers[i]
        for i in range(inst.n_drivers)
        if float(driver_flow[i]) <= 1e-7
    ]
    busy = [
        f"{inst.drivers[i]} ({float(driver_flow[i]):.1f}/{inst.driver_hours[i]:.0f} ч)"
        for i in range(inst.n_drivers)
        if float(driver_flow[i]) > 1e-7
    ]
    lines.append(f"Задействованы: {'; '.join(busy)}.")
    lines.append("")
    if idle:
        lines.append(
            f"Не задействованы: {'; '.join(idle)}. Причины предметные: "
            "окна не совпадают с окнами рейсов и/или ставка выше, чем у альтернатив — "
            "модель выбирает дешёвую совместимую мощность."
        )
        lines.append("")
    lines.append(
        "Дефицитные и стрессовые сценарии показывают: сеть устойчива к одиночному отказу дуги "
        "(перераспределение за счёт запаса окон), но рост спроса на 20% резко поднимает затраты, "
        "так как дешёвая мощность уже исчерпана; критическое сочетание делает задачу несовместимой."
    )
    lines.append("")

    lines.append("## 14. Можно ли реализовать найденное решение на практике")
    lines.append("")
    lines.append(
        "Решение реализуемо как ориентир диспетчеризации, но ограничено моделью: "
        "1) дробные назначения (часть рейса двум водителям) на практике недопустимы — требуется MILP "
        "(см. раздел 15); "
        "2) не моделируются последовательность и пересечения окон при нескольких рейсах одного водителя "
        "в течение дня, а также время на перерывы и возврат в депо; "
        "3) затраты линейны — нет сверхурочных, ночных тарифов, штрафов за опоздание; "
        "4) спрос и время в пути детерминированы: задержки, пробки и отказы техники не учтены; "
        "5) окна времени и подача заданы точно, на практике они оценочны. "
        "Для внедрения нужна проверка последовательности рейсов, целочисленный план и буфер времени."
    )
    lines.append("")

    lines.append("## 15. Дополнительное задание: LP против MILP (неделимые рейсы)")
    lines.append("")
    lines.append(
        f"LP (дробные рейсы): {_num(milp_row['lp_fun'])} руб., время {milp_row['lp_time'] * 1000:.2f} мс; "
        f"MILP (каждый рейс целиком одному водителю, бинарные y_ij, HiGHS branch-and-bound): "
        f"{_num(milp_row['milp_fun'])} руб., статус {milp_row['milp_status']}, "
        f"зазор {milp_row['mip_gap']}, узлов {milp_row['mip_nodes']}, "
        f"время {milp_row['milp_time'] * 1000:.2f} мс."
    )
    lines.append("")
    lines.append(
        f"Зазор неделимости: {milp_row['gap_abs']:.2f} руб. ({milp_row['gap_pct']:.2f}%). "
        f"{milp_row['lp_note']}"
    )
    lines.append("")
    if "methods.png" in figures:
        lines.append(f"![methods.png]({fig_prefix}methods.png)")
        lines.append("")
    lines.append(
        "Вывод: LP-релаксация допускает разделение рейсов и поэтому недооценивает затраты; "
        "MILP даёт реализуемый последовательный план с небольшим (доли процента) удорожанием; "
        "время решения обоих методов сопоставимо и измеряется миллисекундами на масштабе 6×6."
    )
    lines.append("")

    lines.append("## 16. Используемые артефакты")
    lines.append("")
    seed = int(meta["seed"])
    for name, desc in [
        (f"out/seed_{seed}/solution.csv", "таблица найденных потоков"),
        (f"out/seed_{seed}/verification.txt", "результаты независимой проверки"),
        (f"out/seed_{seed}/scenarios.csv", "сценарии и диагностика несовместимости"),
        (f"out/seed_{seed}/shadow_prices.csv", "теневые цены активных ограничений"),
        (f"out/seed_{seed}/demand_duals.csv", "потенциалы рейсов (двойственные переменные спроса)"),
        (f"out/seed_{seed}/greedy_comparison.csv", "сравнение с жадной эвристикой"),
        (f"out/seed_{seed}/milp_comparison.csv", "сравнение LP и MILP"),
        (f"out/seed_{seed}/dual_check.txt", "проверка сильной двойственности"),
        (f"out/seed_{seed}/solver_log.txt", "лог HiGHS"),
        (f"out/seed_{seed}/figures/*.png", "графики и схемы сети"),
        (f"report/seed_{seed}/REPORT.pdf", "отчёт варианта"),
    ]:
        lines.append(f"- `{name}` — {desc}.")
    lines.append("")
    return "\n\n".join(lines) + "\n"


def demand_up_shadows(demand_up: dict) -> list[dict]:
    from .scenarios import shadow_prices

    return shadow_prices(demand_up["result"], demand_up["model"])
