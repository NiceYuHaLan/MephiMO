# Лабораторная работа № 1. Метод линейного программирования

## Что делает

Полный цикл работы с непрерывной линейной задачей оптимизации: генерация данных варианта,
построение модели, решение через HiGHS, независимая проверка допустимости, сценарный анализ
и анализ теневых цен.

Дополнительное задание: сравнение dual simplex (`highs-ds`) и interior point (`highs-ipm`)
по времени и числу итераций; результат сохраняется в `out/seed_<seed>/method_comparison.csv`.

## Индивидуальный вариант

N = 18, Q = 2, Y = 26

V = 1 + ((17 * N + 7 * Q + Y) mod 20) = 7

seed генерации данных = 10000 + 100 * Q + N = 10218

Задача: управление запасами склада на месяц (5 товаров, 4 недели), минимизация затрат на закупку и хранение.

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
- `src/data.py` — генерация данных с seed и чтение из CSV/JSON.
- `src/model.py` — сборка матриц LP-модели.
- `src/solve.py` — решение через HiGHS, сравнение симплекса и IPM.
- `src/verify.py` — независимая проверка на NumPy.
- `src/scenarios.py` — четыре сценария, диагностика infeasible и теневые цены.
- `src/report.py` — CSV-таблицы, графики, отчёт.
- `src/pdf_report.py` — Markdown -> PDF (WeasyPrint, fallback: headless Firefox).
- `src/main.py` — точка входа.

## Требования

- Python 3.11+ (проверено на 3.14.3)
- numpy, pandas, scipy, matplotlib, markdown, weasyprint


