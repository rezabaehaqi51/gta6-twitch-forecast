"""Phase C: greedy combination of Phase A/B winners + robustness checks.

Config = PARAMS overrides + GLOBAL_PARAMS overrides + blend weights +
post-blend spread kwargs (+ optional heat-kernel floor, tested through
a local member builder that mirrors lolo_ens.member_rows with a floor
parameter; if it wins it gets implemented in main.py and re-verified).

Acceptance: union improves >= 0.0005 over the running config and GTA
stays <= 0.1490 (buffer under the 0.1494 gate). Final candidate gets
split-half, jackknife and per-target win checks against v3.
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
POOL = lolo.load_pool(case)
W0 = {"base": 0.65, "heat": 0.18, "smooth": 0.12, "global": 0.05}


def member_rows_floor(pool, idx, params, gparams, floor):
    tgt = pool[idx]
    p1, pre3 = tgt["p1"], tgt["pre3"]
    base_pool = le.subpool(pool, idx, params)
    members = {}
    base = model.log_rows_from_pool(p1, base_pool, params, pre3)
    if base is not None:
        members["base"] = base
    if pre3 is not None:
        kern = []
        for ratios, w, lh1, cpre in base_pool:
            if cpre is not None:
                d = (cpre - pre3) / model.PRE_KERNEL_SIGMA
                w = w * (floor + (1.0 - floor) * math.exp(-0.5 * d * d))
            if w >= params["min_weight"]:
                kern.append((ratios, w, lh1, cpre))
        if len(kern) >= 3:
            rows = model.log_rows_from_pool(p1, kern, params, pre3)
            if rows is not None:
                members["heat"] = rows
    smoothed = [(model.smooth_curve(r, model.SMOOTH_WIDTH), w, lh1, cpre)
                for r, w, lh1, cpre in base_pool]
    rows = model.log_rows_from_pool(p1, smoothed, params, pre3)
    if rows is not None:
        members["smooth"] = rows
    global_pool = le.subpool(pool, idx, gparams)
    if len(global_pool) >= 3:
        rows = model.log_rows_from_pool(p1, global_pool, gparams, pre3)
        if rows is not None:
            members["global"] = rows
    return members


def eval_cfg(cfg):
    params = dict(model.PARAMS)
    params.update(cfg.get("params", {}))
    gparams = dict(model.GLOBAL_PARAMS)
    gparams.update(cfg.get("gparams", {}))
    weights = cfg.get("weights", W0)
    spread_kw = cfg.get("spread_kw", {})
    floor = cfg.get("heat_floor")
    losses, covs = {}, {}
    for idx, tgt in enumerate(POOL):
        if floor is None:
            members = le.member_rows(POOL, idx, params, gparams)
        else:
            members = member_rows_floor(POOL, idx, params, gparams, floor)
        rows = le.blend_rows(members, weights)
        if rows is not None and spread_kw:
            rows = le.apply_spread(rows, **spread_kw)
        losses[tgt["name"]] = le.score_rows(POOL, idx, rows)
        c = le.coverage90(POOL, idx, rows)
        if c is not None:
            covs[tgt["name"]] = c
    return losses, covs


def show(tag, losses):
    s = le.subsets(POOL, losses)
    print(f"{tag:52s} GTA {s['gta']:.4f} big {s['big']:.4f} "
          f"UNION {s['union']:.4f} full {s['full']:.4f}")
    return s


MOVES = [
    ("sp_hi 0.85->0.80", {"params": {"sp_hi": 0.80}}),
    ("blend_normal 0.4->0.3", {"params": {"blend_normal": 0.3}}),
    ("blend -> (0.60,0.18,0.12,0.10)",
     {"weights": {"base": 0.60, "heat": 0.18, "smooth": 0.12,
                  "global": 0.10}}),
    ("blend -> (0.55,0.20,0.15,0.10)",
     {"weights": {"base": 0.55, "heat": 0.20, "smooth": 0.15,
                  "global": 0.10}}),
    ("sp_lo 1.0->1.05", {"params": {"sp_lo": 1.05}}),
    ("tail 1.1->1.2", {"params": {"tail": 1.2}}),
    ("spread 0.7->0.75", {"params": {"spread": 0.75}}),
    ("sp_g2 1.0->1.05", {"params": {"sp_g2": 1.05}}),
    ("post spread x1.03", {"spread_kw": {"global_f": 1.03}}),
    ("post asym lo1.05/hi0.95",
     {"spread_kw": {"lo_f": 1.05, "hi_f": 0.95}}),
    ("heat floor 0.1->0.0", {"heat_floor": 0.0}),
    ("heat floor 0.1->0.2", {"heat_floor": 0.2}),
    ("GLOBAL spread 0.7->0.75", {"gparams": {"spread": 0.75}}),
    ("w_tier_early 0.5->0.6", {"params": {"w_tier_early": 0.6}}),
]


def merge(cfg, move):
    out = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in cfg.items()}
    for k, v in move.items():
        if isinstance(v, dict):
            out.setdefault(k, {}).update(v)
        else:
            out[k] = v
    return out


def main():
    cfg = {}
    losses, _ = eval_cfg(cfg)
    best = show("v3 (running start)", losses)
    for round_no in range(3):
        improved = False
        for tag, move in MOVES:
            trial = merge(cfg, move)
            if trial == cfg:
                continue
            tl, _ = eval_cfg(trial)
            s = le.subsets(POOL, tl)
            if s["union"] < best["union"] - 0.0005 and s["gta"] <= 0.1490:
                print(f"  ACCEPT {tag}: UNION {s['union']:.4f} "
                      f"GTA {s['gta']:.4f}")
                cfg, best, losses = trial, s, tl
                improved = True
        if not improved:
            break
    print("\nfinal config:", json.dumps(cfg, indent=1))
    s = show("FINAL candidate", losses)
    v3_losses, v3_covs = eval_cfg({})
    _, cand_covs = eval_cfg(cfg)

    union_names = [t["name"] for t in POOL
                   if lolo.gta_like(t) or t["p1"] >= 100000]
    # split-half on union targets (alternate by sorted name)
    names_sorted = sorted(union_names)
    for half_tag, half in [("half A", names_sorted[::2]),
                           ("half B", names_sorted[1::2])]:
        a = sum(v3_losses[n] for n in half) / len(half)
        b = sum(losses[n] for n in half) / len(half)
        print(f"split {half_tag}: v3 {a:.4f} cand {b:.4f} "
              f"({'WIN' if b < a else 'LOSE'})")
    # jackknife over union targets
    wins = 0
    worst = 0.0
    for drop in union_names:
        rest = [n for n in union_names if n != drop]
        a = sum(v3_losses[n] for n in rest) / len(rest)
        b = sum(losses[n] for n in rest) / len(rest)
        if b < a:
            wins += 1
        worst = max(worst, b - a)
    print(f"jackknife: cand wins {wins}/{len(union_names)} drops, "
          f"worst margin {worst:+.4f}")
    w = sum(1 for n in union_names if losses[n] < v3_losses[n])
    print(f"per-target: cand wins {w}/{len(union_names)} union targets")
    uc = [cand_covs[n] for n in union_names if n in cand_covs]
    uv = [v3_covs[n] for n in union_names if n in v3_covs]
    print(f"coverage90 union: v3 {sum(uv)/len(uv):.3f} "
          f"cand {sum(uc)/len(uc):.3f}")
    Path("/tmp/tune6c_config.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
