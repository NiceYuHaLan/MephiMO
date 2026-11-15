import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import platform
import scipy

from .model import Instance

SCENARIO_LABELS = {
    "base": "базовый",
    "deficit": "дефицитный",
    "stress": "стрессовый",
    "critical": "критический",
}


def _num(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _top_shifts(base_x, other_x, products, weeks, limit: int = 4) -> list[tuple[float, int, int, float]]:
    shifts = []
    for p in range(base_x.shape[0]):
        for t in range(base_x.shape[1]):
            delta = float(other_x[p, t] - base_x[p, t])
            if abs(delta) > 1e-6:
                shifts.append((abs(delta), p, t, delta))
    shifts.sort(key=lambda item: item[0], reverse=True)
    return shifts[:limit]


def _delta(fun, base):
    if fun is None or base is None:
        return "—", "—"
    delta = float(fun) - float(base)
    pct = delta / float(base) * 100.0 if abs(float(base)) > 1e-12 else float("nan")
    return f"{delta:+.2f}", f"{pct:+.2f}%"


def _short_shadow_label(label: str) -> str:
    if label.startswith("capacity["):
        week = label[len("capacity["):-1].replace("нед.", "нед. ")
        return f"вместимость, {week}"
    if label.startswith("order_limit["):
        week = label[len("order_limit["):-1].replace("нед.", "нед. ")
        return f"лимит закупок, {week}"
    if label.startswith("safety_stock["):
        product = label[len("safety_stock["):-1]
        return f"страховой запас: {product}"
    return label


def write_solution_csv(inst: Instance, ver, path) -> None:
    lines = ["product,week,procurement,inventory"]
    if ver.get("x") is None or ver.get("inventory") is None:
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for p in range(inst.n_products):
        for t in range(inst.n_weeks):
            lines.append(f"{inst.products[p]},{inst.weeks[t]},{ver['x'][p, t]:.6f},{ver['inventory'][p, t]:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verification_txt(ver, path) -> None:
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
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scenario_csv(results: list[dict], path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["scenario", "status", "objective", "feasible", "description", "diagnostics"])
        for r in results:
            fun = r["result"].fun
            writer.writerow([
                r["name"],
                r["result"].status_text,
                "" if fun is None else f"{float(fun):.6f}",
                r["verification"]["feasible"],
                r["description"],
                " | ".join(r.get("diagnostics", [])),
            ])


def write_shadows_csv(shadows: list[dict], path) -> None:
    lines = ["label,type,rhs,marginal,shadow_price,active,predicted_delta_per_unit,actual_delta_per_unit,actual_ok"]
    for s in shadows:
        lines.append(
            f"{s['label']},{s['type']},{s['constraint_rhs']},{s['marginal']},{s['shadow_price']},{int(s['active'])},"
            f"{s['predicted_delta_per_unit']},{s['actual_delta_per_unit']},{int(s['actual_ok'])}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_csv(rows: list[dict], path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["method", "status", "objective", "iterations", "time_median_s", "time_min_s", "repeats"])
        for r in rows:
            writer.writerow([
                r["method"],
                r["status"],
                "" if r["objective"] is None else f"{float(r['objective']):.6f}",
                "" if r["nit"] is None else int(r["nit"]),
                f"{r['time_median']:.6f}",
                f"{r['time_min']:.6f}",
                r["repeats"],
            ])


def make_figures(scenario_results: list[dict], shadows: list[dict], base_ver, inst: Instance, fig_dir) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    P, T = base_ver["x"].shape

    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(T)
    width = 0.8 / P
    for p in range(P):
        ax.bar(idx + p * width, base_ver["x"][p], width, label=f"товар {inst.products[p]}")
    ax.set_xticks(idx + width * (P - 1) / 2)
    ax.set_xticklabels([f"нед. {w}" for w in inst.weeks])
    ax.set_ylabel("Закупка, ед.")
    ax.set_title("План закупок по продуктам")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    out = fig_dir / "plan_purchases.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    made.append(out.name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in range(P):
        ax.plot(
            list(range(0, T + 1)),
            [float(inst.initial_stock[p])] + [float(v) for v in base_ver["inventory"][p]],
            marker="o",
            label=inst.products[p],
        )
    ax.set_xticks(list(range(0, T + 1)))
    ax.set_xticklabels(["начало"] + [f"нед. {w}" for w in inst.weeks])
    ax.set_ylabel("Запас, ед.")
    ax.set_title("Динамика запасов")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    out = fig_dir / "inventory.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    made.append(out.name)

    names = [r["name"] for r in scenario_results]
    fun = [r["result"].fun if r["result"].fun is not None else 0.0 for r in scenario_results]
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3"][: len(names)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(np.arange(len(names)), fun, color=colors)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([SCENARIO_LABELS.get(n, n) for n in names])
    ax.set_ylabel("Целевая функция, руб.")
    ax.set_title("Сравнение сценариев")
    for i, r in enumerate(scenario_results):
        if r["result"].fun is None:
            ax.text(i, 0.0, "infeasible", ha="center", va="bottom", fontsize=8, color="#C44E52")
        else:
            ax.text(i, fun[i], f"{fun[i]:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = fig_dir / "scenarios.png"
    fig.savefig(out)
    plt.close(fig)
    made.append(out.name)

    active_rows = sorted([s for s in shadows if s["active"]], key=lambda s: s["shadow_price"])
    if active_rows:
        labels = [_short_shadow_label(s["label"]) for s in active_rows]
        vals = [s["shadow_price"] for s in active_rows]
        color_by_type = {"capacity": "#4C72B0", "order_limit": "#55A868", "safety_stock": "#DD8452"}
        colors = [color_by_type.get(s["type"], "#8172B3") for s in active_rows]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.barh(labels, vals, color=colors)
        ax.set_xlabel("Теневая цена, руб. на +1 ед. RHS")
        ax.set_title("Теневые цены активных ограничений")
        for bar, value in zip(bars, vals):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=8,
            )
        ax.margins(x=0.18)
        fig.tight_layout()
        out = fig_dir / "shadow.png"
        fig.savefig(out)
        plt.close(fig)
        made.append(out.name)

    return made


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_report(
    scenario_results: list[dict],
    shadows: list[dict],
    base_ver,
    inst: Instance,
    meta: dict,
    fig_prefix: str = "../out/figures/",
    figures: list[str] | None = None,
    method_rows: list[dict] | None = None,
) -> str:
    P, T = inst.n_products, inst.n_weeks
    sol_rows = []
    if base_ver["x"] is not None and base_ver["inventory"] is not None:
        for p in range(P):
            for t in range(T):
                sol_rows.append(
                    [
                        inst.products[p],
                        str(inst.weeks[t]),
                        f"{base_ver['x'][p, t]:.2f}",
                        f"{base_ver['inventory'][p, t]:.2f}",
                    ]
                )

    def scenario(name: str):
        return next((r for r in scenario_results if r["name"] == name), None)

    base_scenario = scenario("base") or scenario_results[0]
    deficit_scenario = scenario("deficit")
    stress_scenario = scenario("stress")
    base_obj = base_scenario["result"].fun

    scenario_rows = []
    for r in scenario_results:
        delta, pct = _delta(r["result"].fun, base_obj)
        scenario_rows.append([
            r["name"],
            r["result"].status_text,
            _num(r["result"].fun),
            delta,
            pct,
            str(r["verification"]["feasible"]),
        ])

    active_shadow_rows = [
        [
            _short_shadow_label(s["label"]),
            f"{s['constraint_rhs']:.2f}",
            f"{s['shadow_price']:.4f}",
            f"{s['predicted_delta_per_unit']:.4f}",
            f"{s['actual_delta_per_unit']:.4f}",
            str(s["actual_ok"]),
        ]
        for s in shadows
        if s["active"]
    ]
    active = [s for s in shadows if s["active"]]
    active_labels = ", ".join(_short_shadow_label(s["label"]) for s in active)

    lines = []
    lines.append("# Отчёт по лабораторной работе № 1. Метод линейного программирования")
    lines.append("")
    lines.append("## 1. Постановка задачи и индивидуальный вариант")
    lines.append("")
    lines.append(
        f"Число варианта: V = {meta['variant']}; студент N = {meta['N']}; группа Q = {meta['Q']}; год Y = {meta['Y']}; seed = {meta['seed']}."
    )
    lines.append("")
    lines.append(
        f"Задача: управление запасами склада на месяц. Нужно построить непрерывную линейную модель "
        f"и минимизировать затраты на закупку и хранение при распределении {inst.n_products} товаров по {inst.n_weeks} неделям."
    )
    lines.append("")
    lines.append("## 2. Данные варианта и процедура генерации")
    lines.append("")
    lines.append(
        f"Данные варианта сгенерированы детерминированно при seed = {meta['seed']}: спрос товаров — целые "
        f"значения в диапазоне 40..120 ед./нед.; базовая цена закупки — равномерно в диапазоне 80..160 руб./ед. "
        f"с недельной вариацией ±8%; стоимость хранения — 2% от базовой цены; начальный запас — целое в диапазоне 10..19 ед.; "
        f"вместимость склада — max(30% от пикового недельного совокупного спроса, ceil(Σ начального запаса / 0.8)) ед.; "
        f"лимит закупок — 150% от пикового недельного совокупного спроса; страховой запас — 10% от среднего спроса."
    )
    lines.append("")
    lines.append(_md_table(["Параметр", "Единица измерения"], [
        ["d[p,t]", "ед./нед."],
        ["c[p,t]", "руб./ед."],
        ["h_p", "руб./(ед.·нед.)"],
        ["x[p,t], I[p,t]", "ед."],
        ["W_t", "ед."],
        ["O_t", "ед./нед."],
        ["S_p", "ед."],
        ["Целевая функция", "руб."],
    ]))
    lines.append("")
    lines.append("## 3. Формальная модель")
    lines.append("")
    lines.append(f"Множества: P = {{{', '.join(inst.products)}}}, T = {{{', '.join(str(w) for w in inst.weeks)}}}.")
    lines.append("")
    lines.append("Переменные: x[p,t] ≥ 0 — объём закупки товара p на неделе t, ед.; I[p,t] ≥ 0 — запас на конец недели t, ед.")
    lines.append("")
    lines.append(
        "Параметры: d[p,t] — спрос, ед./нед.; c[p,t] — цена закупки, руб./ед.; h_p — стоимость хранения, "
        "руб./(ед.·нед.); I0[p] — начальный запас, ед.; W_t — вместимость склада, ед.; O_t — лимит закупок, "
        "ед./нед.; S_p — страховой запас, ед."
    )
    lines.append("")
    lines.append("min cᵀx + hᵀI")
    lines.append("")
    lines.append("- Для каждого товара p ∈ P и недели t ∈ T выполняется баланс: I[p,t] − I[p,t−1] − x[p,t] = −d[p,t], при I[p,−1] = I0[p].")
    lines.append("- Для каждой недели t ∈ T: Σ_p I[p,t] ≤ W_t (вместимость).")
    lines.append("- Для каждой недели t ∈ T: Σ_p x[p,t] ≤ O_t (лимит закупок).")
    lines.append("- Для каждого товара p ∈ P: I[p,T−1] ≥ S_p (страховой запас).")
    lines.append("- x[p,t] ≥ 0, I[p,t] ≥ 0.")
    lines.append("")
    lines.append("## 4. Допущения и учебные упрощения")
    lines.append("")
    lines.append("Детерминированный спрос; линейные издержки; фиксированные цены и лимиты; отсутствие сроков поставки, дефицита и скидок. Вместимость `W_t` — жёсткий лимит на суммарный запас на конец недели; генерация гарантирует, что начальный запас помещается в `W_1` и в `0.8·W_1`. Товары измеряются в сопоставимых условных единицах; стоимость хранения начисляется на запас на конец недели, начальный запас отдельно не тарифицируется; страховой запас требуется на конец горизонта.")
    lines.append("")
    lines.append("## 5. Реализация")
    lines.append("")
    lines.append("Модель собирается из матриц A_eq/A_ub и решается через SciPy `linprog` с методом HiGHS, options: presolve=True, disp=False. Базовый сценарий решается методом dual simplex (`highs-ds`); в дополнительном задании сравниваются `highs-ds` и `highs-ipm`.")
    lines.append(f"Версии: Python {platform.python_version()}, SciPy {scipy.__version__}; HiGHS поставляется в составе SciPy, версии зафиксированы в requirements.txt.")
    lines.append("")
    lines.append("Интерфейс: `build_model(Instance) → LPModel(c, A_ub, b_ub, A_eq, b_eq, bounds, constraint_labels, equality_labels, index_x, index_i)`.")
    lines.append("")
    lines.append(f"Задача решена за seed {meta['seed']}: статус {base_scenario['result'].status_text}, целевая функция {_num(base_obj)} руб., итераций {base_scenario['result'].nit}.")
    lines.append("")
    lines.append("## 6. Статус решения и интерпретация")
    lines.append("")
    lines.append(f"Статус: {base_scenario['result'].status_text}. Целевая функция: {_num(base_obj)} руб.")
    lines.append("")
    if base_scenario["result"].status_text == "optimal":
        lines.append(
            "Статус `optimal` означает, что найдено конечное допустимое решение: все ограничения выполнены "
            "с допуском 1e-7, целевая функция достигает минимума. Статусы `infeasible` и `unbounded` означали бы "
            "несовместность ограничений или неограниченность задачи; для них требуется отдельная диагностика."
        )
        lines.append("В текущем запуске лимит времени не задан; при наличии time limit статус `time limit` означал бы остановку до доказательства оптимальности.")
    lines.append("")
    lines.append(f"Итераций HiGHS: {base_scenario['result'].nit}. Активные ограничения: {active_labels or 'нет'}.")
    lines.append("")
    lines.append("## 7. Независимая проверка")
    lines.append("")
    lines.append(_md_table(["check", "ok", "violation"], [[c["name"], str(c["ok"]), f"{c['violation']:.3e}"] for c in base_ver["checks"]]))
    lines.append("")
    objective_diff = base_ver["objective_diff"]
    objective_diff_text = f"{objective_diff:.3e}" if objective_diff is not None else "—"
    lines.append(f"Objective из солвера: {_num(base_ver['objective_solver'])}; пересчитанный из x и I средствами NumPy: {_num(base_ver['objective_numpy'])}; разница: {objective_diff_text}.")
    lines.append("")
    lines.append("## 8. Сценарный анализ")
    lines.append("")
    lines.append(_md_table(["scenario", "status", "objective", "ΔC", "Δ%", "feasible"], scenario_rows))
    lines.append("")
    feasible_scenarios = [r for r in scenario_results if r["result"].status == 0 and r["verification"]["x"] is not None]
    if feasible_scenarios:
        weekly_headers = ["scenario"] + [f"закупки нед. {w}" for w in inst.weeks] + ["совокупные"]
        weekly_rows = [
            [r["name"]]
            + [f"{float(r['verification']['x'][:, t].sum()):.1f}" for t in range(inst.n_weeks)]
            + [f"{float(r['verification']['x'].sum()):.1f}"]
            for r in feasible_scenarios
        ]
        lines.append(_md_table(weekly_headers, weekly_rows))
        lines.append("")
    lines.append(
        "Сценарии: "
        + "; ".join(
            f"{SCENARIO_LABELS.get(r['name'], r['name'])} — {r['description'].rstrip('.')}"
            for r in scenario_results
        )
        + "."
    )
    lines.append("")
    if deficit_scenario is not None and base_ver["x"] is not None and deficit_scenario["verification"]["x"] is not None:
        d_delta, d_pct = _delta(deficit_scenario["result"].fun, base_obj)
        shifts = _top_shifts(base_ver["x"], deficit_scenario["verification"]["x"], inst.products, inst.weeks)
        shift_text = "; ".join(
            f"{inst.products[p]}, нед. {inst.weeks[t]}: {base_ver['x'][p, t]:.0f} → {deficit_scenario['verification']['x'][p, t]:.0f}"
            for _, p, t, _ in shifts
        )
        lines.append(f"Дефицитный сценарий: ΔC = {d_delta} руб. ({d_pct}). Наибольшие изменения плана: {shift_text}.")
    elif deficit_scenario is not None:
        d_delta, d_pct = _delta(deficit_scenario["result"].fun, base_obj)
        lines.append(f"Дефицитный сценарий: ΔC = {d_delta} руб. ({d_pct}).")
    if stress_scenario is not None and base_ver["x"] is not None and stress_scenario["verification"]["x"] is not None:
        s_delta, s_pct = _delta(stress_scenario["result"].fun, base_obj)
        shifts = _top_shifts(base_ver["x"], stress_scenario["verification"]["x"], inst.products, inst.weeks)
        shift_text = "; ".join(
            f"{inst.products[p]}, нед. {inst.weeks[t]}: {base_ver['x'][p, t]:.0f} → {stress_scenario['verification']['x'][p, t]:.0f}"
            for _, p, t, _ in shifts
        )
        lines.append("")
        lines.append(f"Стрессовый сценарий: ΔC = {s_delta} руб. ({s_pct}). Одновременное снижение вместимости и рост спроса усиливают дефицит складских мощностей. Наибольшие изменения плана: {shift_text}.")
    elif stress_scenario is not None:
        s_delta, s_pct = _delta(stress_scenario["result"].fun, base_obj)
        lines.append("")
        lines.append(f"Стрессовый сценарий: ΔC = {s_delta} руб. ({s_pct}). Одновременное снижение вместимости и рост спроса усиливают дефицит складских мощностей.")
    lines.append("")
    cap_shadow = max((s for s in shadows if s["type"] == "capacity"), key=lambda s: s["shadow_price"], default=None)
    if (
        cap_shadow is not None
        and deficit_scenario is not None
        and deficit_scenario["result"].fun is not None
        and base_obj is not None
    ):
        try:
            week_idx = int(str(cap_shadow["label"]).split("нед.")[-1].rstrip("]")) - 1
        except (ValueError, IndexError):
            week_idx = 0
        drop = float(inst.capacity[week_idx] - deficit_scenario["instance"].capacity[week_idx])
        predicted = cap_shadow["shadow_price"] * drop
        actual = float(deficit_scenario["result"].fun - base_obj)
        lines.append(
            f"Сверка теневой цены: для {_short_shadow_label(cap_shadow['label'])} теневая цена {cap_shadow['shadow_price']:.2f} руб./ед.; "
            f"снижение вместимости на {drop:.0f} ед. даёт прогноз {predicted:.2f} руб. и факт {actual:+.2f} руб. "
            f"Различие объясняется тем, что теневая цена локальна и при большом изменении RHS базис меняется."
        )
        lines.append("")
    for r in scenario_results:
        if r["result"].status != 0:
            lines.append(f"Диагностика сценария `{r['name']}` (статус {r['result'].status_text}):")
            if r.get("diagnostics"):
                for issue in r["diagnostics"]:
                    lines.append(f"- {issue}")
            else:
                lines.append("- автоматическая диагностика не выявила простых конфликтов; требуется анализ двойственной задачи.")
            lines.append("")
    lines.append("## 9. Теневые цены")
    lines.append("")
    lines.append(_md_table(["constraint", "rhs", "shadow", "predicted_delta", "actual_delta", "actual_ok"], active_shadow_rows))
    lines.append("")
    lines.append("В таблице показаны только активные ограничения; неактивные имеют нулевую теневую цену, полный список — в `shadow_prices.csv`.")
    lines.append("")
    lines.append(
        "На графике положительное значение означает снижение затрат при увеличении RHS на 1 ед. "
        "Для страховых запасов используется RHS = −S_p, поэтому это стоимость снижения норматива на 1 ед.; "
        "неактивные ограничения на график не нанесены."
    )
    lines.append("")
    if active:
        lines.append("Активные ограничения: " + active_labels + ".")
        lines.append("")
        cap_active = [s for s in active if s["type"] == "capacity"]
        if cap_active:
            best_cap = max(cap_active, key=lambda s: s["shadow_price"])
            lines.append(
                f"Из ресурсов в первую очередь выгодно наращивать {_short_shadow_label(best_cap['label'])}: +1 ед. вместимости снижает "
                f"затраты на {best_cap['shadow_price']:.2f} руб. Остальные недели вместимости не активны, поэтому "
                f"дополнительные площади там не дают эффекта."
            )
        order_active = [s for s in active if s["type"] == "order_limit"]
        if not order_active:
            lines.append("Лимиты закупок не активны: их увеличение в текущем плане не уменьшает затраты.")
        safety_active = [s for s in active if s["type"] == "safety_stock"]
        if safety_active:
            cheapest = min(safety_active, key=lambda s: s["shadow_price"])
            lines.append(
                f"Страховые запасы заданы как I[p,T−1] ≥ S_p, поэтому в матрице используется b = −S_p. "
                f"Положительная теневая цена — это стоимость единицы норматива; самый дешёвый норматив — "
                f"{_short_shadow_label(cheapest['label'])} ({cheapest['shadow_price']:.2f} руб./ед.), самый дорогой — "
                f"{_short_shadow_label(max(safety_active, key=lambda s: s['shadow_price'])['label'])} "
                f"({max(safety_active, key=lambda s: s['shadow_price'])['shadow_price']:.2f} руб./ед.)."
            )
    else:
        lines.append("Активных ограничений с ненулевой теневой ценой нет.")
    lines.append("")
    lines.append("## 10. Содержательный вывод")
    lines.append("")
    lines.append(f"Оптимальная совокупная стоимость за месяц составила {_num(base_obj)} руб. (закупки {_num(base_ver['purchase_part'])} + хранение {_num(base_ver['holding_part'])}).")
    lines.append("")
    if sol_rows:
        lines.append(_md_table(["Товар", "Неделя", "Закупка, ед.", "Запас, ед."], sol_rows))
        lines.append("")
    zero_notes = []
    if base_ver["x"] is not None and base_ver["inventory"] is not None:
        for p in range(P):
            for t in range(T):
                if abs(float(base_ver["x"][p, t])) <= 1e-7:
                    carry = float(inst.initial_stock[p]) if t == 0 else float(base_ver["inventory"][p, t - 1])
                    tail = f" и страховой запас {inst.safety_stock[p]:.0f} ед" if t == T - 1 else ""
                    zero_notes.append(
                        f"{inst.products[p]}, нед. {inst.weeks[t]}: запас на начало {carry:.0f} ед. покрывает спрос {inst.demand[p, t]:.0f} ед.{tail}"
                    )
    if zero_notes:
        lines.append("Нулевые закупки: " + "; ".join(zero_notes) + ". В оптимуме покупка в эти недели невыгодна: запас покрывает спрос, а альтернативная закупка даёт больший учёт хранения.")
    else:
        lines.append("Нулевых закупок в оптимальном плане нет.")
    lines.append("")
    if (
        base_ver["x"] is not None
        and base_ver["inventory"] is not None
        and np.allclose(base_ver["x"], np.round(base_ver["x"]))
        and np.allclose(base_ver["inventory"], np.round(base_ver["inventory"]))
    ):
        lines.append("Найденный план оказался целочисленным, хотя модель непрерывная; это удобно для практики, но не гарантируется для других данных.")
        lines.append("")
    lines.append("## 11. Практическая реализуемость")
    lines.append("")
    lines.append(
        "План можно использовать как ориентир, но его практическая реализация ограничена: модель не учитывает "
        "дискретность партий и неделимую упаковку, сроки поставки и возможные задержки, неопределённость спроса "
        "и необходимость пересчёта плана при отклонениях, а также нелинейные тарифы хранения. Страховой запас "
        "задан жёстким ограничением, поэтому модель не оценивает компромисс между уровнем сервиса и затратами. "
        "Для внедрения нужны буферы, правила перепланирования и проверка плана на целочисленность. Кроме того, "
        "вместимость `W_t` является жёсткой для суммарного запаса на конец недели; генерация гарантирует, что начальный "
        "запас помещается в `W_1` и в `0.8·W_1`, поэтому дефицитный и стрессовый сценарии не нарушают эту физическую "
        "проверку. Для внедрения также необходимо фиксировать соглашение о единицах измерения разных товаров и правила "
        "резервирования."
    )
    lines.append("")
    lines.append("## 12. Дополнительное задание: симплекс vs внутренняя точка")
    lines.append("")
    if method_rows:
        method_table = [
            [
                r["method"],
                r["status"],
                _num(r["objective"]),
                "—" if r["nit"] is None else str(int(r["nit"])),
                f"{r['time_median'] * 1000:.2f}",
                f"{r['time_min'] * 1000:.2f}",
            ]
            for r in method_rows
        ]
        lines.append(_md_table(["method", "status", "objective", "iterations", "median_time_ms", "min_time_ms"], method_table))
        lines.append("")
        lines.append(
            "Сравнение выполнено на одной и той же модели с `presolve=True`; для каждого метода выполнено 3 запуска "
            "и взято медианное время. `highs-ds` — dual simplex, `highs-ipm` — interior point."
        )
        lines.append("")
        objectives = [r["objective"] for r in method_rows if r["objective"] is not None]
        if len(objectives) >= 2 and max(objectives) - min(objectives) < 1e-6:
            lines.append(
                "Оба метода нашли одно и то же оптимальное значение целевой функции. Время и число итераций "
                "различаются; счётчики итераций симплекса и IPM имеют разный смысл и напрямую не сравнимы."
            )
        else:
            lines.append("Значения целевой функции или статусы методов различаются; требуется дополнительный анализ.")
        lines.append("")
    else:
        lines.append("Сравнение методов не выполнялось.")
        lines.append("")
    lines.append("## 13. Графики")
    lines.append("")
    fig_list = figures if figures is not None else ["plan_purchases.png", "inventory.png", "scenarios.png", "shadow.png"]
    if fig_list:
        for name in fig_list:
            lines.append(f"![{name}]({fig_prefix}{name})")
    else:
        lines.append("Графики не сформированы.")
    lines.append("")
    lines.append("## 14. Используемые артефакты")
    lines.append("")
    lines.append("- `out/seed_" + str(int(meta['seed'])) + "/solution.csv` - таблица решения.")
    lines.append("- `out/seed_" + str(int(meta['seed'])) + "/verification.txt` - результаты проверки.")
    lines.append("- `out/seed_" + str(int(meta['seed'])) + "/scenarios.csv` - сценарии, включая диагностику infeasible.")
    lines.append("- `out/seed_" + str(int(meta['seed'])) + "/shadow_prices.csv` - теневые цены.")
    lines.append("- `out/seed_" + str(int(meta['seed'])) + "/method_comparison.csv` - сравнение симплекса и IPM.")
    lines.append("- `out/seed_" + str(int(meta['seed'])) + "/solver_log.txt` - лог HiGHS.")
    lines.append("- `out/seed_" + str(int(meta['seed'])) + "/figures/*.png` - графики.")
    lines.append("- `report/seed_" + str(int(meta['seed'])) + "/REPORT.pdf` - отчёт варианта.")
    lines.append("")
    return "\n\n".join(lines) + "\n"
