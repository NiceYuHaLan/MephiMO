from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
REPORT_DIR = ROOT / "report"

N = 18
Q = 2
Y = 26

TOL = 1e-7

TRAVEL_H_PER_ZONE = 0.25


@dataclass(frozen=True)
class Paths:
    data: Path
    out: Path
    fig: Path
    report: Path


def variant(n: int = N, q: int = Q, y: int = Y) -> tuple[int, int]:
    v = 1 + ((17 * n + 7 * q + y) % 20)
    seed = 10000 + 100 * q + n
    return v, seed


def paths_for(seed: int) -> Paths:
    base = OUT_DIR / f"seed_{seed}"
    return Paths(
        data=DATA_DIR / f"seed_{seed}",
        out=base,
        fig=base / "figures",
        report=REPORT_DIR / f"seed_{seed}",
    )
