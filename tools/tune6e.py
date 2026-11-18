"""Phase C3: last probes around C1. Adopt only a clear win:
union -0.001 or better vs C1 with GTA no worse than C1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lolo_ens as le  # noqa: E402
import tune6c  # noqa: E402
import tune6d  # noqa: E402,F401  (prints the C-table on import is fine)

POOL = tune6c.POOL
BL = {"base": 0.60, "heat": 0.18, "smooth": 0.12, "global": 0.10}
ASYM = {"lo_f": 1.05, "hi_f": 0.95}
C1 = {"weights": BL, "params": {"sp_lo": 1.05}, "spread_kw": ASYM}

PROBES = {
    "P1 C1 + global member 0.12": {
        "weights": {"base": 0.58, "heat": 0.18, "smooth": 0.12,
                    "global": 0.12},
        "params": {"sp_lo": 1.05}, "spread_kw": ASYM},
    "P2 C1 + sp_lo 1.08": {
        "weights": BL, "params": {"sp_lo": 1.08}, "spread_kw": ASYM},
    "P3 C1 + post hi 0.93": {
        "weights": BL, "params": {"sp_lo": 1.05},
        "spread_kw": {"lo_f": 1.05, "hi_f": 0.93}},
    "P4 C1 + post lo 1.08": {
        "weights": BL, "params": {"sp_lo": 1.05},
        "spread_kw": {"lo_f": 1.08, "hi_f": 0.95}},
    "P5 C1 + sp_g2 1.05": {
        "weights": BL, "params": {"sp_lo": 1.05, "sp_g2": 1.05},
        "spread_kw": ASYM},
    "P6 C1 + blend_normal 0.3": {
        "weights": BL, "params": {"sp_lo": 1.05, "blend_normal": 0.3},
        "spread_kw": ASYM},
    "P7 C1 + heat 0.20/base 0.58": {
        "weights": {"base": 0.58, "heat": 0.20, "smooth": 0.12,
                    "global": 0.10},
        "params": {"sp_lo": 1.05}, "spread_kw": ASYM},
}

for tag, cfg in PROBES.items():
    losses, covs = tune6c.eval_cfg(cfg)
    s = le.subsets(POOL, losses)
    print(f"{tag:32s} GTA {s['gta']:.4f} big {s['big']:.4f} "
          f"UNION {s['union']:.4f} full {s['full']:.4f}")
