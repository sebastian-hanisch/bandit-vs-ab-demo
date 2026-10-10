"""Erzeugt `precomputed_pools.json` (je Sicherheitsfaktor 10 000 unabhängige Standort-Fenster, SplitMix64; dieselben Pools wie im
ab-test-demo) und `precomputed_sweeps.json` (Sweeps über Effekt, Horizont, Trend und Rundengröße mit je 3 000 Wiederholungen).

    python generate_precomputed.py            # Pools etwa eine Minute, Sweeps einige Minuten
    python generate_precomputed.py --sweeps   # nur die Sweeps (Pools müssen vorhanden sein)"""

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import bva_constants as C
import bva_evaluation as E
from bva_rng import SplitMix64
from bva_vehicle import window_service_level

POOLS_OUT = Path(__file__).resolve().parent / "precomputed_pools.json"
SWEEPS_OUT = Path(__file__).resolve().parent / "precomputed_sweeps.json"
SWEEP_REPS = 3000
DEFAULTS = {"T": C.DEFAULT_T, "m": C.DEFAULT_M, "trend": C.DEFAULT_TREND}

# Zellen der Sweeps: (Name, z, T, m, Trend)
CELLS = (
    [("effect", z, C.DEFAULT_T, C.DEFAULT_M, 0.0) for z in (1.0, 1.2, 1.4, 1.6, 1.8)]
    + [("horizon", z, t, C.DEFAULT_M, 0.0) for z in (1.0, 1.2, 1.4) for t in (150, 300, 600, 900)]
    + [("trend", z, C.DEFAULT_T, C.DEFAULT_M, tr) for z in (1.0, 1.4) for tr in (0.0, 0.01, 0.02, 0.03, 0.04)]
    + [("round", z, C.DEFAULT_T, m, 0.0) for z in (1.0, 1.4) for m in (5, 10, 25, 50)]
)


def unit_seed(z_index, i):
    return 9_000_011 * (z_index + 1) + 7 * i + 1


def _chunk(args):
    z_index, z, start, stop = args
    return [round(window_service_level(z, SplitMix64(unit_seed(z_index, i))), 6) for i in range(start, stop)]


def make_pools():
    step = 500
    jobs = [(zi, z, s, min(s + step, C.POOL_SIZE)) for zi, z in enumerate(C.Z_GRID) for s in range(0, C.POOL_SIZE, step)]
    with mp.Pool() as pool:
        parts = pool.map(_chunk, jobs)
    pools = {f"{z:.1f}": [] for z in C.Z_GRID}
    for (zi, z, _s, _e), values in zip(jobs, parts):
        pools[f"{z:.1f}"].extend(values)
    meta = {"pool_size": C.POOL_SIZE, "window_days": C.WINDOW_DAYS, "burn_days": C.BURN_DAYS, "base_mean": C.BASE_MEAN,
            "disp": C.DISP, "lead": C.LEAD, "z_grid": list(C.Z_GRID)}
    POOLS_OUT.write_text(json.dumps({"meta": meta, "pools": pools}, separators=(",", ":")), encoding="utf-8")


_POOLS = None


def _cell(args):
    global _POOLS
    if _POOLS is None:
        _POOLS = E.load_pools()
    name, z, t, m, trend = args
    s = E.summarize(_POOLS, z, t, m, trend, reps=SWEEP_REPS, tag=3)
    return {"sweep": name, "z": z, "T": t, "m": m, "trend": trend, "reps": SWEEP_REPS, "means": s["means"],
            "truth": s["truth"], "rows": s["rows"]}


def make_sweeps():
    with mp.Pool() as pool:
        cells = pool.map(_cell, CELLS, chunksize=1)
    meta = {"reps": SWEEP_REPS, "floor": C.FLOOR, "test_frac": C.TEST_FRAC, "alpha": C.ALPHA}
    SWEEPS_OUT.write_text(json.dumps({"meta": meta, "cells": cells}, separators=(",", ":")), encoding="utf-8")


def main():
    t0 = time.time()
    if "--sweeps" not in sys.argv:
        make_pools()
        print(f"{POOLS_OUT.name}: {len(C.Z_GRID)} Pools à {C.POOL_SIZE}, {time.time() - t0:.0f} s", file=sys.stderr)
    make_sweeps()
    print(f"{SWEEPS_OUT.name}: {len(CELLS)} Zellen à {SWEEP_REPS} Wiederholungen, {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
