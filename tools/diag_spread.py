"""Diagnostics: which union targets gain/lose from post-blend widening?

Prints per-target v3 loss, loss at global spread x1.10, delta, hour-1
size, and base-pool effective sample size (ESS) -- to see whether size
or pool richness separates the gainers from the losers.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model  # noqa: E402
import lolo  # noqa: E402
import lolo_ens as le  # noqa: E402

case = json.loads((lolo.KIT_INPUT / "p00.in").read_text())
pool = lolo.load_pool(case)
W = {"base": 0.65, "heat": 0.18, "smooth": 0.12, "global": 0.05}

rows_out = []
for idx, tgt in enumerate(pool):
    if not (lolo.gta_like(tgt) or tgt["p1"] >= 100000):
        continue
    members = le.member_rows(pool, idx)
    rows = le.blend_rows(members, W)
    l0 = le.score_rows(pool, idx, rows)
    l1 = le.score_rows(pool, idx, le.apply_spread(rows, global_f=1.10))
    sub = le.subpool(pool, idx, model.PARAMS)
    sw = sum(w for _, w, _, _ in sub)
    ess = sw * sw / sum(w * w for _, w, _, _ in sub) if sub else 0.0
    rows_out.append((tgt["name"], tgt["p1"], lolo.gta_like(tgt), ess,
                     l0, l1, l1 - l0))

rows_out.sort(key=lambda r: r[6])
print(f"{'target':28s} {'p1':>9s} {'gta':>4s} {'ESS':>5s} "
      f"{'v3':>7s} {'x1.10':>7s} {'delta':>7s}")
for name, p1, gta, ess, l0, l1, d in rows_out:
    print(f"{name:28s} {p1:9.0f} {str(gta):>4s} {ess:5.1f} "
          f"{l0:7.4f} {l1:7.4f} {d:+7.4f}")
