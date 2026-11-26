# Лабораторная работа № 2. Транспортные и сетевые модели

## Что делает

Полный цикл работы с транспортной моделью «Назначение водителей на рейсы с окнами времени»:
генерация данных варианта, построение LP-модели, решение через HiGHS, независимая проверка
баланса верщин и границ дуг на NumPy, схема сети с ёмкостями и потоками (NetworkX),
сценарный анализ, анализ узких мест и теневых цен (потенциалов), сравнение с жадной
эвристикой.

Дополнительное задание: сравнение LP (дробные рейсы) и MILP (неделимые рейсы) через
HiGHS; результат сохраняется в `out/seed_<seed>/milp_comparison.csv`.

## Индивидуальный вариант

N = 18, Q = 2, Y = 26

V = 1 + ((17 * N + 7 * Q + Y) mod 20) = 7

seed генерации данных = 10000 + 100 * Q + N = 10218

Задача: 6 водителей с часовыми окнами и почасовыми ставками должны покрыть 6 рейсов
(длительность, спрос, окно выполнения). Дуга (водитель, рейс) существует, если после
подачи и между рейсового простоя остаётся достаточно времени в окне водителя.
Минимизация суммарной стоимости: min Σ c_ij·x_ij, где c_ij = rate_i·(1 + t_ij/h_j).

## Запуск

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Запуск:

```bash
python3 -m src.main
```

Разные варианты:

```bash
python3 -m src.main --n 5 --q 0 --y 26
python3 -m src.main --n 1 --q 1 --y 26
```

Перегенерация данных:

```bash
python3 -m src.main --regenerate
```

## Структура

- `src/config.py` — константы, вариант, пути (`paths_for`).
- `src/data.py` — генерация данных с seed, валидация, чтение/запись CSV/JSON.
- `src/model.py` — сборка LP-модели (переменные, ограничения, метки).
- `src/solve.py` — решение через HiGHS (`highs-ds`), MILP (`milp`), лог решателя.
- `src/verify.py` — независимая проверка баланса/границ на NumPy, двойственность (RC/CS).
- `src/scenarios.py` — сценарии (перегрузка, рост спроса +20%, отказ дуги, стресс,
  критический), диагностика infeasible по условию Холла.
- `src/greedy.py` — жадная эвристика назначения для сравнения.
- `src/report.py` — CSV-таблицы, схемы сети (NetworkX), графики, отчёт (Markdown).
- `src/pdf_report.py` — Markdown → PDF (WeasyPrint, fallback: headless Firefox).
- `src/main.py` — точка входа.

## Артефакты

- `data/` — входные CSV/JSON (`drivers.csv`, `trips.csv`, `arcs.csv`, `meta.json`).
- `out/seed_<seed>/` — `solution.csv`, `verification.txt`, `scenarios.csv`,
  `shadow_prices.csv`, `demand_duals.csv`, `greedy_comparison.csv`,
  `milp_comparison.csv`, `dual_check.txt`, `solver_log.txt`, `figures/*.png`.
- `report/seed_<seed>/` — `REPORT.md` и `REPORT.pdf`.

## Требования

- Python 3.11+ (проверено на 3.14.3)
- numpy, pandas, scipy, matplotlib, networkx, markdown, weasyprint
