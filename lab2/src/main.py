import argparse
import os
import tempfile

import numpy as np

from . import config
from . import pdf_report
from . import report as rep
from .data import ensure_data
from .greedy import greedy_cheapest_first
from .model import build_model
from .scenarios import demand_duals, run_scenarios, shadow_prices
from .solve import log_text, solve, solve_milp
from .verify import dual_check, verify, verify_milp


def main(n: int = 18, q: int = 2, y: int = 26, regenerate: bool = False) -> None:
    V, seed = config.variant(n, q, y)

    paths = config.paths_for(seed)
    paths.data.mkdir(parents=True, exist_ok=True)
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.fig.mkdir(parents=True, exist_ok=True)
    paths.report.mkdir(parents=True, exist_ok=True)

    inst, meta = ensure_data(seed, paths.data, regenerate)

    model = build_model(inst)
    with tempfile.TemporaryFile(mode="w+") as solver_tmp:
        old_fd_1 = os.dup(1)
        old_fd_2 = os.dup(2)
        try:
            os.dup2(solver_tmp.fileno(), 1)
            os.dup2(solver_tmp.fileno(), 2)
            result = solve(model, disp=True)
            solver_tmp.seek(0)
            solver_raw = solver_tmp.read()
        finally:
            os.dup2(old_fd_1, 1)
            os.dup2(old_fd_2, 2)
            os.close(old_fd_1)
            os.close(old_fd_2)

    solver_log = log_text(result, model, inst, f"ЛР № 2 base seed={seed}") + "\n=== HiGHS output ===\n" + solver_raw
    verification = verify(inst, model, result)
    dual = dual_check(inst, model, result)
    shadows = shadow_prices(result, model)
    d_duals = demand_duals(result, model)

    base_x = verification["x"] if verification["x"] is not None else np.zeros(inst.n_arcs)
    scenario_results = run_scenarios(inst, base_x)

    solver_log += "\n=== Статусы сценариев ===\n"
    for r in scenario_results:
        fun = r["result"].fun
        solver_log += (
            f"{r['name']}: {r['result'].status_text}, objective="
            f"{'None' if fun is None else f'{fun:.6f}'}, nit={r['result'].nit}\n"
        )
    (paths.out / "solver_log.txt").write_text(solver_log, encoding="utf-8")

    greedy_rows = []
    for r in scenario_results:
        if r["result"].status != 0:
            greedy_rows.append({
                "scenario": r["name"],
                "lp_status": r["result"].status_text,
                "lp_cost": None,
                "greedy_cost": None,
                "greedy_unmet": float(np.sum(r["instance"].trip_hours)),
                "greedy_saturated": 0,
            })
            continue
        g = greedy_cheapest_first(r["instance"])
        greedy_rows.append({
            "scenario": r["name"],
            "lp_status": r["result"].status_text,
            "lp_cost": r["result"].fun,
            "greedy_cost": g["cost"],
            "greedy_unmet": g["unmet_h"],
            "greedy_saturated": len(g["saturated_arcs"]),
        })

    milp = solve_milp(inst)
    milp_ver = verify_milp(inst, milp.y, milp.fun) if milp.y is not None else {"feasible": False}
    lp_note_parts = []
    if verification["x"] is not None:
        split = [
            inst.trips[j]
            for j in range(inst.n_trips)
            if sum(
                1
                for k in range(inst.n_arcs)
                if inst.arc_trip[k] == j and verification["x"][k] > 1e-7
            ) > 1
        ]
        if split:
            lp_note_parts.append(f"LP разделяет рейсы между водителями: {', '.join(split)}")
        else:
            lp_note_parts.append("LP-решение оказалось целым по рейсам")
    base_greedy = next(g for g in greedy_rows if g["scenario"] == "base")
    milp_row = {
        "lp_fun": result.fun,
        "lp_time": result.solve_time_s,
        "milp_fun": milp.fun,
        "milp_time": milp.solve_time_s,
        "milp_status": milp.status_text,
        "mip_gap": milp.mip_gap,
        "mip_nodes": milp.mip_node_count,
        "gap_abs": (milp.fun - result.fun) if milp.fun is not None else float("nan"),
        "gap_pct": ((milp.fun - result.fun) / result.fun * 100.0) if milp.fun is not None else float("nan"),
        "lp_note": "; ".join(lp_note_parts),
        "milp_note": (
            f"MIP gap {milp.mip_gap}, узлов {milp.mip_node_count}, "
            f"независимая проверка: {'PASS' if milp_ver.get('feasible') else 'FAIL'}"
        ),
        "greedy_fun": base_greedy["greedy_cost"],
    }

    rep.write_solution_csv(inst, verification, paths.out / "solution.csv")
    rep.write_verification_txt(verification, paths.out / "verification.txt")
    rep.write_scenario_csv(scenario_results, result.fun, paths.out / "scenarios.csv")
    rep.write_shadows_csv(shadows, paths.out / "shadow_prices.csv")
    rep.write_demand_duals_csv(d_duals, paths.out / "demand_duals.csv")
    rep.write_greedy_csv(greedy_rows, paths.out / "greedy_comparison.csv")
    rep.write_milp_csv(milp_row, paths.out / "milp_comparison.csv")
    rep.write_dual_check_txt(dual, paths.out / "dual_check.txt")

    figures = rep.make_figures(
        inst,
        verification,
        scenario_results,
        shadows,
        paths.fig,
        greedy_rows,
        milp_row,
        base_x,
    )

    report_text = rep.build_report(
        inst,
        meta,
        verification,
        dual,
        scenario_results,
        shadows,
        d_duals,
        greedy_rows,
        milp_row,
        figures,
        f"../../out/seed_{seed}/figures/",
    )
    (paths.report / "REPORT.md").write_text(report_text, encoding="utf-8")
    ok = pdf_report.render(report_text, paths.report / "REPORT.pdf", base_url=str(paths.report))
    print(f"PDF отчет (seed {seed}, V={V}): {'сгенерирован' if ok else 'не сгенерирован'} -> {paths.report / 'REPORT.pdf'}")
    print(f"Вариант: V={V}, seed={seed}")
    print(f"LP: {result.status_text}, objective={result.fun:.2f} руб.")
    print(f"MILP: {milp.status_text}, objective={milp.fun} руб.")
    for r in scenario_results:
        fun = r["result"].fun
        print(f"  сценарий {r['name']}: {r['result'].status_text}" + ("" if fun is None else f", objective={fun:.2f}"))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Лабораторная работа № 2: транспортные и сетевые модели.")
    parser.add_argument("--n", type=int, default=18, help="порядковый номер студента в группе N")
    parser.add_argument("--q", type=int, default=2, help="индекс учебной группы Q")
    parser.add_argument("--y", type=int, default=26, help="две последние цифры года Y")
    parser.add_argument("--regenerate", action="store_true", help="перегенерировать данные с фиксированным seed")
    args = parser.parse_args()
    main(n=args.n, q=args.q, y=args.y, regenerate=args.regenerate)


if __name__ == "__main__":
    cli()
