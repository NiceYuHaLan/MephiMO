import argparse
import os
import tempfile

from . import config
from . import pdf_report
from . import report as rep
from .data import ensure_data
from .model import build_model
from .scenarios import run_scenarios, shadow_prices
from .solve import compare_methods, log_text, solve
from .verify import verify


def main(n: int = 18, q: int = 2, y: int = 26, regenerate: bool = False) -> None:
    V, seed = config.variant(n, q, y)

    paths = config.paths_for(seed)
    paths.data.mkdir(parents=True, exist_ok=True)
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.report.mkdir(parents=True, exist_ok=True)

    inst, meta = ensure_data(seed, paths, regenerate, n=n, q=q, y=y)
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
    solver_log = log_text(result, model, f"ЛР № 1 base seed={seed}") + "\n=== HiGHS output ===\n" + solver_raw
    (paths.out / "solver_log.txt").write_text(solver_log, encoding="utf-8")
    verification = verify(inst, model, result)
    shadows = shadow_prices(model, result)
    scenario_results = run_scenarios(inst)
    method_rows = compare_methods(model)

    rep.write_solution_csv(inst, verification, paths.out / "solution.csv")
    rep.write_verification_txt(verification, paths.out / "verification.txt")
    rep.write_scenario_csv(scenario_results, paths.out / "scenarios.csv")
    rep.write_shadows_csv(shadows, paths.out / "shadow_prices.csv")
    rep.write_method_csv(method_rows, paths.out / "method_comparison.csv")
    figures = rep.make_figures(scenario_results, shadows, verification, inst, paths.fig)
    report_text = rep.build_report(
        scenario_results,
        shadows,
        verification,
        inst,
        meta,
        f"../../out/seed_{seed}/figures/",
        figures=figures,
        method_rows=method_rows,
    )
    ok = pdf_report.render(report_text, paths.report / "REPORT.pdf", base_url=str(paths.report))
    print(f"PDF отчет (seed {seed}, V={V}): {'сгенерирован' if ok else 'не сгенерирован'} -> {paths.report / 'REPORT.pdf'}")
    print(f"Вариант: V={V}, seed={seed}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Лабораторная работа № 1 по линейному программированию.")
    parser.add_argument("--n", type=int, default=18, help="порядковый номер студента в группе N")
    parser.add_argument("--q", type=int, default=2, help="индекс учебной группы Q")
    parser.add_argument("--y", type=int, default=26, help="две последние цифры года Y")
    parser.add_argument("--regenerate", action="store_true", help="перегенерировать данные с фиксированным seed")
    args = parser.parse_args()
    main(n=args.n, q=args.q, y=args.y, regenerate=args.regenerate)


if __name__ == "__main__":
    cli()
