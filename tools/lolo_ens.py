"""LOLO harness for the SHIPPED ensemble in ../main.py.

tools/lolo.py scores a single parameter set through forecast_from_pool;
the shipped program is the four-member ensemble in main.ensemble_rows.
This harness scores exactly that: for each of the 37 practice-pool
launches it rebuilds the base and global sub-pools from the other 36
(same weights, same min_weight gate as main.build_pool), calls
main.ensemble_rows, formats with main.format_rows and scores with
lolo.pinball.

It also caches the four members' log-quantile rows per target so blend
weights and post-blend spread calibration can be swept cheaply, and it
asserts the cached-member replica reproduces main.ensemble_rows
bit-for-bit at the shipped configuration before trusting the cache.

Usage:
    python3 tools/lolo_ens.py            # shipped v3 baseline numbers
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model  # noqa: E402
import lolo  # noqa: E402

TARGET = {"console_only": True}


def subpool(pool, idx, params):
    tgt = pool[idx]
    out = []
    for j, c in enumerate(pool):
        if j == idx:
            continue
        w = model.comp_weight(c["feat"], TARGET, c["log_h1"],
                              tgt["log_h1"], params)
        if w < params["min_weight"]:
            continue
        out.append((c["ratios"], w, c["log_h1"], c["pre3"]))
    return out


def member_rows(pool, idx, params=None, gparams=None):
    """The four ensemble members' log rows for target idx (or None)."""
    params = model.PARAMS if params is None else params
    gparams = model.GLOBAL_PARAMS if gparams is None else gparams
    tgt = pool[idx]
    p1, pre3 = tgt["p1"], tgt["pre3"]
    base_pool = subpool(pool, idx, params)
    members = {}
    base = model.log_rows_from_pool(p1, base_pool, params, pre3)
    if base is not None:
        members["base"] = base
    if pre3 is not None:
        kern = []
        for ratios, w, lh1, cpre in base_pool:
            if cpre is not None:
                d = (cpre - pre3) / model.PRE_KERNEL_SIGMA
                w = w * (0.1 + 0.9 * math.exp(-0.5 * d * d))
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
    global_pool = subpool(pool, idx, gparams)
    if len(global_pool) >= 3:
        rows = model.log_rows_from_pool(p1, global_pool, gparams, pre3)
        if rows is not None:
            members["global"] = rows
    return members


def blend_rows(members, weights):
    """weights: {member_key: w}; renormalised over present members."""
    present = [(k, weights[k]) for k in weights if k in members]
    if not present:
        return None
    total = sum(w for _, w in present)
    return [[sum(w * members[k][i][j] for k, w in present) / total
             for j in range(5)] for i in range(23)]


def apply_spread(rows, global_f=1.0, blocks=None, lo_f=1.0, hi_f=1.0,
                 tail_f=1.0):
    """Post-blend spread calibration around each hour's median (log space)."""
    if global_f == 1.0 and not blocks and lo_f == 1.0 and hi_f == 1.0 \
            and tail_f == 1.0:
        return rows
    out = []
    for i, qs in enumerate(rows):
        h = i + 2
        f = global_f
        if blocks:
            if h <= 5:
                f *= blocks[0]
            elif h <= 12:
                f *= blocks[1]
            elif h <= 18:
                f *= blocks[2]
            else:
                f *= blocks[3]
        med = qs[2]
        adj = []
        for j, q in enumerate(qs):
            d = (q - med) * f
            if j < 2:
                d *= lo_f
            elif j > 2:
                d *= hi_f
            if j in (0, 4):
                d *= tail_f
            adj.append(med + d)
        out.append(adj)
    return out


def score_rows(pool, idx, rows):
    tgt = pool[idx]
    if rows is None:
        fc = model.baseline_forecast(tgt["p1"])
    else:
        fc = model.format_rows(tgt["p1"], rows)
    return lolo.pinball(fc, tgt["actual"])


def coverage90(pool, idx, rows):
    """Fraction of scored hours whose actual lies inside [q05, q95]."""
    tgt = pool[idx]
    if rows is None:
        return None
    inside, n = 0, 0
    for i, qs in enumerate(rows):
        y = tgt["actual"].get(i + 2)
        if not y or y <= 0:
            continue
        ly = math.log(y) - math.log(tgt["p1"])
        n += 1
        if qs[0] <= ly <= qs[4]:
            inside += 1
    return inside / n if n else None


def subsets(pool, losses):
    def mean(sel):
        vals = [losses[t["name"]] for t in pool if sel(t)]
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "gta": mean(lolo.gta_like),
        "big": mean(lambda t: t["p1"] >= 100000),
        "union": mean(lambda t: lolo.gta_like(t) or t["p1"] >= 100000),
        "full": mean(lambda t: True),
    }


def post_kw():
    """The shipped post-blend calibration, as apply_spread kwargs."""
    if hasattr(model, "POST_LO"):
        return {"lo_f": model.POST_LO, "hi_f": model.POST_HI}
    return {}


def evaluate_ensemble(pool, params=None, gparams=None, weights=None,
                      spread_kw=None):
    """Full LOLO of the ensemble under the given configuration."""
    weights = weights or {"base": model.BLEND[0], "heat": model.BLEND[1],
                          "smooth": model.BLEND[2],
                          "global": model.BLEND[3]}
    kw = post_kw()
    kw.update(spread_kw or {})
    losses, covs = {}, {}
    for idx, tgt in enumerate(pool):
        members = member_rows(pool, idx, params, gparams)
        rows = blend_rows(members, weights)
        if rows is not None and kw:
            rows = apply_spread(rows, **kw)
        losses[tgt["name"]] = score_rows(pool, idx, rows)
        c = coverage90(pool, idx, rows)
        if c is not None:
            covs[tgt["name"]] = c
    return losses, covs


def verify_replica(pool):
    """Cached members + BLEND must equal main.ensemble_rows exactly."""
    weights = {"base": model.BLEND[0], "heat": model.BLEND[1],
               "smooth": model.BLEND[2], "global": model.BLEND[3]}
    for idx in (0, 5, 12):
        tgt = pool[idx]
        direct = model.ensemble_rows(tgt["p1"], subpool(pool, idx,
                                     model.PARAMS),
                                     subpool(pool, idx,
                                             model.GLOBAL_PARAMS),
                                     tgt["pre3"])
        replica = blend_rows(member_rows(pool, idx), weights)
        if replica is not None and post_kw():
            replica = apply_spread(replica, **post_kw())
        assert (direct is None) == (replica is None), idx
        if direct is not None:
            for i in range(23):
                for j in range(5):
                    assert abs(direct[i][j] - replica[i][j]) < 1e-12, \
                        (idx, i, j)
    print("replica check: member cache == main.ensemble_rows (exact)")


def main():
    case = json.loads((lolo.KIT_INPUT / "p00.in").read_text())
    pool = lolo.load_pool(case)
    print(f"pool: {len(pool)} launches "
          f"({sum(1 for t in pool if lolo.gta_like(t))} GTA-like, "
          f"{sum(1 for t in pool if t['p1'] >= 100000)} big)")
    verify_replica(pool)
    losses, covs = evaluate_ensemble(pool)
    s = subsets(pool, losses)
    ucov = [covs[t["name"]] for t in pool
            if lolo.gta_like(t) or t["p1"] >= 100000]
    print(f"shipped ensemble LOLO: GTA {s['gta']:.4f} "
          f"big {s['big']:.4f} UNION {s['union']:.4f} "
          f"full {s['full']:.4f}")
    print(f"union coverage90 (per-target mean): "
          f"{sum(ucov) / len(ucov):.3f}")


if __name__ == "__main__":
    main()
