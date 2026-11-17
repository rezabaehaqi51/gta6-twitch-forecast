"""Union-objective tuning: GTA-family OR big launches as the target set.

GTA VI is both family-correct (console, rolling, single-player) and
almost surely big (its category printed 1.42M viewers for a premiere
event), so candidate configs are judged on the union of the two LOLO
subsets (21 launches), with the components still reported.

Knobs searched here, on top of tune2.forecast_x:
  * tilt_cap: cap the size tilt's extrapolation distance
    |ln p1 - xbar| at a multiple of the pool's size sd.
  * family blend: mix the full-pool forecast with a forecast built
    only from GTA-family pool members.
  * the tune3 ensemble partners (top-10 local, smoothed, global).
  * a final global spread multiplier on the blended result.

Usage: python3 tools/tune5.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as model  # noqa: E402
import tune2  # noqa: E402
import tune3  # noqa: E402

LEVELS = model.LEVELS
Z = model.Z


def fam(t):
    f = t["feat"]
    return (f["console_only"] and f["tier"] == "comparable"
            and not f["simultaneous_only"] and not f["early_access"])


def forecast_variant(p1, entries, params, tx, tilt_cap=None):
    """tune2.forecast_x with an optional cap on the size-tilt distance."""
    if tilt_cap is None:
        return tune2.forecast_x(p1, entries, params, tx)
    # Monkey-see: recompute with capped tilt by shrinking tilt_shrink is
    # not equivalent; implement the cap by clipping the target's log
    # size inside a shallow copy of the pipeline: emulate via params is
    # impossible, so do a direct variant here.
    return _forecast_capped(p1, entries, params, tx, tilt_cap)


def _forecast_capped(p1, entries, params, tx, cap_mult):
    log_p1 = math.log(p1)
    rows = []
    for h in range(2, 25):
        sample = [(r[h], w, lh1, pre3, vpc, psl)
                  for r, w, lh1, pre3, vpc, psl in entries if h in r]
        if len(sample) < 3:
            rows.append(None)
            continue
        sample.sort(key=lambda t: t[0])
        pairs = [(v, w) for v, w, _, _, _, _ in sample]
        wsum = sum(w for _, w, _, _, _, _ in sample)
        mean = sum(v * w for v, w, _, _, _, _ in sample) / wsum
        var = sum(w * (v - mean) ** 2 for v, w, _, _, _, _ in sample) / wsum
        sd = math.sqrt(max(var, 1e-6))
        xbar = sum(lh1 * w for _, w, lh1, _, _, _ in sample) / wsum
        sxx = sum(w * (lh1 - xbar) ** 2 for _, w, lh1, _, _, _ in sample)
        sxy = sum(w * (lh1 - xbar) * (v - mean)
                  for v, w, lh1, _, _, _ in sample)
        slope = (sxy / sxx) if sxx > 1e-9 else 0.0
        xsd = math.sqrt(sxx / wsum) if wsum else 0.0
        dist = log_p1 - xbar
        cap = cap_mult * xsd
        if dist > cap:
            dist = cap
        elif dist < -cap:
            dist = -cap
        centre = mean + params["tilt_shrink"] * slope * dist
        if params["pre_shrink"] and tx.get("pre3") is not None:
            pts = [(s[3], s[0], s[1]) for s in sample if s[3] is not None]
            if len(pts) >= 6:
                tw = sum(w for _, _, w in pts)
                px = sum(x * w for x, _, w in pts) / tw
                py = sum(y * w for _, y, w in pts) / tw
                psxx = sum(w * (x - px) ** 2 for x, _, w in pts)
                psxy = sum(w * (x - px) * (y - py) for x, y, w in pts)
                if psxx > 1e-9:
                    centre += params["pre_shrink"] * (psxy / psxx) \
                        * (tx["pre3"] - px)
        qs = []
        for tau, z in zip(LEVELS, Z):
            emp = model.weighted_quantile(pairs, tau)
            emp_c = centre + (emp - mean)
            norm = centre + z * sd
            qs.append((1.0 - params["blend_normal"]) * emp_c
                      + params["blend_normal"] * norm)
        rows.append(qs)
    known_rows = [r for r in rows if r is not None]
    if not known_rows:
        return None
    mean_row = [sum(r[i] for r in known_rows) / len(known_rows)
                for i in range(5)]
    for i, r in enumerate(rows):
        if r is None:
            left = rows[i - 1] if i > 0 else None
            right = next((rows[j] for j in range(i + 1, 23)
                          if rows[j] is not None), None)
            if left is not None and right is not None:
                rows[i] = [(a + b) / 2.0 for a, b in zip(left, right)]
            elif left is not None:
                rows[i] = list(left)
            elif right is not None:
                rows[i] = list(right)
            else:
                rows[i] = list(mean_row)
    out = []
    for i, qs in enumerate(rows):
        h = i + 2
        med = qs[2]
        gm = tune2._group_mult(h, params)
        adj = []
        for j, q in enumerate(qs):
            dd = (q - med) * params["spread"] * gm
            if j < 2:
                dd *= params["sp_lo"]
            elif j > 2:
                dd *= params["sp_hi"]
            if j in (0, 4):
                dd *= params["tail"]
            adj.append(med + dd)
        vals = []
        prev = 0.0
        for lq in adj:
            v = p1 * math.exp(lq)
            v = min(max(v, 1.0), 1_000_000_000.0)
            v = max(v, prev)
            vals.append(round(v, 2))
            prev = v
        out.append({"hour": h, "q": vals})
    return {"hours": out}


def entries_for(pool, tgt, params, family_only=False):
    out = []
    for c in pool:
        if c["name"] == tgt["name"]:
            continue
        if family_only and not fam(c):
            continue
        w = model.comp_weight(c["feat"], {"console_only": True},
                              c["log_h1"], tgt["log_h1"], params)
        if w < params["min_weight"]:
            continue
        out.append((c["ratios"], w, c["log_h1"], c["pre3"], c["vpc"],
                    c["pre_slope"]))
    return out


def subsets_u(pool, losses):
    def mean(sel):
        vals = [losses[t["name"]] for t in pool if sel(t)]
        return sum(vals) / len(vals) if vals else float("nan")
    gta = mean(tune2.gta_like)
    big = mean(lambda t: t["p1"] >= 100000)
    union = mean(lambda t: tune2.gta_like(t) or t["p1"] >= 100000)
    full = mean(lambda t: True)
    return {"gta": gta, "big": big, "union": union, "full": full}


def main():
    case = json.loads((tune2.KIT_INPUT / "p00.in").read_text())
    pool = tune2.load_pool_x(case)
    base = tune3.V2_PARAMS

    def eval_config(params, tilt_cap=None, fam_w=0.0):
        losses = {}
        for tgt in pool:
            tx = {"pre3": tgt["pre3"], "vpc": tgt["vpc"],
                  "pre_slope": tgt["pre_slope"]}
            fc = forecast_variant(tgt["p1"],
                                  entries_for(pool, tgt, params),
                                  params, tx, tilt_cap)
            if fc is None:
                fc = model.baseline_forecast(tgt["p1"])
            lq = tune3.lq_rows(fc)
            if fam_w:
                ef = entries_for(pool, tgt, params, family_only=True)
                if len(ef) >= 3:
                    ff = forecast_variant(tgt["p1"], ef, params, tx,
                                          tilt_cap)
                    if ff is not None:
                        lf = tune3.lq_rows(ff)
                        lq = {h: [(1 - fam_w) * a + fam_w * b
                                  for a, b in zip(lq[h], lf[h])]
                              for h in lq}
            losses[tgt["name"]] = tune3.pinball_lq(lq, tgt["actual"])
        return losses

    def show(tag, losses):
        s = subsets_u(pool, losses)
        print(f"{tag:34s} GTA {s['gta']:.4f} big {s['big']:.4f} "
              f"UNION {s['union']:.4f} full {s['full']:.4f}")
        return s

    show("v2 (champion)", eval_config(base))
    for cap in (1.0, 1.5, 2.0):
        show(f"v2 + tilt_cap {cap}", eval_config(base, tilt_cap=cap))
    for fw in (0.15, 0.25):
        show(f"v2 + family blend {fw}", eval_config(base, fam_w=fw))
    show("v2 + cap 1.5 + family 0.15",
         eval_config(base, tilt_cap=1.5, fam_w=0.15))
    for tilt in (0.6, 0.7):
        p = dict(base)
        p["tilt_shrink"] = tilt
        show(f"tilt {tilt} + cap 1.5 + family 0.15",
             eval_config(p, tilt_cap=1.5, fam_w=0.15))


if __name__ == "__main__":
    main()
