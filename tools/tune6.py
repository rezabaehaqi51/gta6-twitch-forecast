"""v4 variant search on the shipped ensemble (tools/lolo_ens.py harness).

Phase A: blend weights + post-blend spread calibration, swept cheaply
over cached member rows (v3 parameters).
Phase B: parameter knobs, each a full ensemble LOLO (~0.4 s).

Adoption gate (from the v4 brief): UNION must improve on 0.1996 and
GTA-family must not worsen beyond 0.1444 + 0.005. Every variant is
printed with both numbers; results also land in /tmp/tune6_results.json.

Usage: python3 tools/tune6.py [A|B|combine]
"""

import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model  # noqa: E402
import lolo  # noqa: E402
import lolo_ens as le  # noqa: E402

V3 = {"union": 0.1996, "gta": 0.1444}
RESULTS = []


def record(tag, losses, covs=None):
    s = le.subsets(POOL, losses)
    row = {"tag": tag, "gta": round(s["gta"], 4), "big": round(s["big"], 4),
           "union": round(s["union"], 4), "full": round(s["full"], 4)}
    if covs:
        ucov = [covs[t["name"]] for t in POOL
                if lolo.gta_like(t) or t["p1"] >= 100000
                and t["name"] in covs]
        row["cov90"] = round(sum(ucov) / len(ucov), 3) if ucov else None
    gate = row["union"] < V3["union"] and row["gta"] <= V3["gta"] + 0.005
    row["gate"] = gate
    RESULTS.append(row)
    print(f"{tag:44s} GTA {row['gta']:.4f} big {row['big']:.4f} "
          f"UNION {row['union']:.4f} full {row['full']:.4f}"
          f"{'  GATE-PASS' if gate else ''}")
    return row


def losses_from_members(members_list, weights, spread_kw=None):
    losses, covs = {}, {}
    for idx, tgt in enumerate(POOL):
        rows = le.blend_rows(members_list[idx], weights)
        if rows is not None and spread_kw:
            rows = le.apply_spread(rows, **spread_kw)
        losses[tgt["name"]] = le.score_rows(POOL, idx, rows)
        c = le.coverage90(POOL, idx, rows)
        if c is not None:
            covs[tgt["name"]] = c
    return losses, covs


def w4(base, heat, smooth, glob):
    return {"base": base, "heat": heat, "smooth": smooth, "global": glob}


def phase_a():
    members_list = [le.member_rows(POOL, i) for i in range(len(POOL))]
    record("A0 v3 blend (0.65/0.18/0.12/0.05)",
           *losses_from_members(members_list, w4(0.65, 0.18, 0.12, 0.05)))
    # single members, diagnostics
    for k in ("base", "heat", "smooth", "global"):
        record(f"A single member: {k}",
               *losses_from_members(members_list,
                                    w4(*(1.0 if kk == k else 0.0
                                         for kk in ("base", "heat",
                                                    "smooth", "global")))))
    blends = [
        (0.60, 0.20, 0.15, 0.05), (0.70, 0.15, 0.10, 0.05),
        (0.65, 0.18, 0.12, 0.00), (0.65, 0.12, 0.18, 0.05),
        (0.65, 0.24, 0.06, 0.05), (0.55, 0.20, 0.15, 0.10),
        (0.60, 0.18, 0.12, 0.10), (0.70, 0.18, 0.12, 0.00),
        (0.75, 0.13, 0.07, 0.05), (0.60, 0.25, 0.10, 0.05),
        (0.65, 0.18, 0.17, 0.00), (0.50, 0.22, 0.18, 0.10),
        (0.65, 0.23, 0.12, 0.00), (0.65, 0.13, 0.12, 0.10),
        (0.58, 0.22, 0.14, 0.06), (0.62, 0.20, 0.13, 0.05),
        (0.68, 0.16, 0.11, 0.05), (0.65, 0.20, 0.10, 0.05),
        (0.65, 0.16, 0.14, 0.05), (0.63, 0.19, 0.12, 0.06),
    ]
    for b in blends:
        record(f"A blend {b}", *losses_from_members(members_list, w4(*b)))
    # post-blend spread at the v3 blend
    for f in (0.90, 0.94, 0.97, 1.03, 1.06, 1.10, 1.15):
        record(f"A spread global x{f}",
               *losses_from_members(members_list, w4(0.65, 0.18, 0.12, 0.05),
                                    {"global_f": f}))
    for tag, blocks in [("early-wide", (1.08, 1.0, 1.0, 1.0)),
                        ("late-wide", (1.0, 1.0, 1.05, 1.05)),
                        ("early+late", (1.06, 1.0, 1.0, 1.06)),
                        ("mid-narrow", (1.0, 0.95, 0.95, 1.0)),
                        ("g1-only-narrow", (0.94, 1.0, 1.0, 1.0))]:
        record(f"A spread blocks {tag}",
               *losses_from_members(members_list, w4(0.65, 0.18, 0.12, 0.05),
                                    {"blocks": blocks}))
    for lo_f, hi_f in [(0.95, 1.0), (1.05, 1.0), (1.0, 0.95), (1.0, 1.05),
                       (0.95, 1.05), (1.05, 0.95)]:
        record(f"A spread asym lo {lo_f} hi {hi_f}",
               *losses_from_members(members_list, w4(0.65, 0.18, 0.12, 0.05),
                                    {"lo_f": lo_f, "hi_f": hi_f}))
    for tf in (0.95, 1.05, 1.10):
        record(f"A spread tail x{tf}",
               *losses_from_members(members_list, w4(0.65, 0.18, 0.12, 0.05),
                                    {"tail_f": tf}))


def phase_b():
    base_params = dict(model.PARAMS)
    base_g = dict(model.GLOBAL_PARAMS)
    knobs = [
        ("size_sigma", [1.25, 1.75, 2.0]), ("blend_normal", [0.3, 0.5]),
        ("spread", [0.65, 0.75, 0.8]), ("tail", [1.0, 1.2]),
        ("tilt_shrink", [0.4, 0.6]), ("pre_shrink", [0.4, 0.6, 0.65]),
        ("w_pc", [0.25, 0.4]), ("w_simultaneous", [0.03, 0.1]),
        ("size_floor", [0.05, 0.1]), ("sp_hi", [0.8, 0.9]),
        ("sp_lo", [0.95, 1.05]), ("sp_g1", [1.1, 1.2]),
        ("sp_g2", [0.95, 1.05]), ("sp_g3", [0.8, 0.9]),
        ("sp_g4", [0.8, 0.9]), ("min_weight", [0.03, 0.08]),
        ("w_tier_early", [0.4, 0.6]), ("w_tier_weak", [0.4, 0.6]),
    ]
    for key, vals in knobs:
        for v in vals:
            p = dict(base_params)
            p[key] = v
            losses, covs = le.evaluate_ensemble(POOL, params=p)
            record(f"B PARAMS {key}={v}", losses, covs)
    gknobs = [
        ("blend_normal", [0.4, 0.6]), ("size_sigma", [3.0, 5.0]),
        ("pre_shrink", [0.25, 0.5]), ("w_pc", [0.45, 0.65]),
        ("size_floor", [0.2, 0.4]), ("spread", [0.65, 0.75]),
        ("w_simultaneous", [0.15, 0.35]),
    ]
    for key, vals in gknobs:
        for v in vals:
            g = dict(base_g)
            g[key] = v
            losses, covs = le.evaluate_ensemble(POOL, gparams=g)
            record(f"B GLOBAL {key}={v}", losses, covs)
    for v in (0.7, 0.85, 1.25, 1.5):
        old = model.PRE_KERNEL_SIGMA
        model.PRE_KERNEL_SIGMA = v
        try:
            losses, covs = le.evaluate_ensemble(POOL)
            record(f"B PRE_KERNEL_SIGMA={v}", losses, covs)
        finally:
            model.PRE_KERNEL_SIGMA = old
    for v in (3, 7):
        old = model.SMOOTH_WIDTH
        model.SMOOTH_WIDTH = v
        try:
            losses, covs = le.evaluate_ensemble(POOL)
            record(f"B SMOOTH_WIDTH={v}", losses, covs)
        finally:
            model.SMOOTH_WIDTH = old


if __name__ == "__main__":
    case = json.loads((lolo.KIT_INPUT / "p00.in").read_text())
    POOL = lolo.load_pool(case)
    phase = sys.argv[1] if len(sys.argv) > 1 else "A"
    if phase == "A":
        phase_a()
    elif phase == "B":
        phase_b()
    out = Path("/tmp/tune6_results.json")
    prev = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(prev + RESULTS, indent=1))
    passed = [r for r in RESULTS if r["gate"]]
    print(f"\n{len(RESULTS)} variants, {len(passed)} gate-pass:")
    for r in sorted(passed, key=lambda r: r["union"]):
        print(f"  {r['tag']:44s} UNION {r['union']:.4f} GTA {r['gta']:.4f}")
