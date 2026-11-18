"""Phase C2: curated config comparison around the greedy candidate.

The full greedy stack costs GTA +0.0040 of its +0.005 budget, mostly
from the PARAMS spread 0.70->0.75 step (union -0.0006 for GTA +0.0023).
Here the subsets are compared head-to-head with the same robustness
checks, to pick the stack with the best gain per unit of GTA cost.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lolo  # noqa: E402
import lolo_ens as le  # noqa: E402
import tune6c  # noqa: E402

POOL = tune6c.POOL
BL = {"base": 0.60, "heat": 0.18, "smooth": 0.12, "global": 0.10}
ASYM = {"lo_f": 1.05, "hi_f": 0.95}

CONFIGS = {
    "C0 v3": {},
    "C1 blend+sp_lo+asym": {"weights": BL, "params": {"sp_lo": 1.05},
                            "spread_kw": ASYM},
    "C2 C1+spread0.72": {"weights": BL,
                         "params": {"sp_lo": 1.05, "spread": 0.72},
                         "spread_kw": ASYM},
    "C3 blend+asym": {"weights": BL, "spread_kw": ASYM},
    "C4 greedy full": {"weights": BL,
                       "params": {"sp_lo": 1.05, "spread": 0.75},
                       "spread_kw": ASYM},
    "C5 blend+sp_lo+spread0.75": {"weights": BL,
                                  "params": {"sp_lo": 1.05,
                                             "spread": 0.75}},
    "C6 sp_lo+asym+spread0.75 (v3 blend)":
        {"params": {"sp_lo": 1.05, "spread": 0.75}, "spread_kw": ASYM},
    "C7 C1+tail post x1.05": {"weights": BL, "params": {"sp_lo": 1.05},
                              "spread_kw": {"lo_f": 1.05, "hi_f": 0.95,
                                            "tail_f": 1.05}},
    "C8 blend+sp_lo+asym-lo-only": {"weights": BL,
                                    "params": {"sp_lo": 1.05},
                                    "spread_kw": {"lo_f": 1.05}},
}

union_names = [t["name"] for t in POOL
               if lolo.gta_like(t) or t["p1"] >= 100000]
names_sorted = sorted(union_names)
v3_losses, _ = tune6c.eval_cfg({})

for tag, cfg in CONFIGS.items():
    losses, covs = tune6c.eval_cfg(cfg)
    s = le.subsets(POOL, losses)
    ha = sum(losses[n] for n in names_sorted[::2]) / len(names_sorted[::2])
    hb = sum(losses[n] for n in names_sorted[1::2]) / len(names_sorted[1::2])
    va = sum(v3_losses[n] for n in names_sorted[::2]) / len(names_sorted[::2])
    vb = sum(v3_losses[n] for n in names_sorted[1::2]) / len(names_sorted[1::2])
    jk = sum(1 for drop in union_names
             if sum(losses[n] for n in union_names if n != drop)
             < sum(v3_losses[n] for n in union_names if n != drop))
    w = sum(1 for n in union_names if losses[n] < v3_losses[n])
    uc = [covs[n] for n in union_names if n in covs]
    print(f"{tag:36s} GTA {s['gta']:.4f} big {s['big']:.4f} "
          f"UNION {s['union']:.4f} full {s['full']:.4f} | "
          f"halves {'W' if ha < va else 'L'}/{'W' if hb < vb else 'L'} "
          f"jk {jk}/{len(union_names)} wins {w}/{len(union_names)} "
          f"cov {sum(uc)/len(uc):.3f}")
