"""Ensemble + shape-model research for the GTA VI forecaster.

Precomputes LOLO forecasts of several diverse forecasters over the 37
practice-pool launches, then searches ensemble blends (log-quantile
space) and a per-hour spread-calibration layer on top. The judge is the
same robust objective as tune2.py: GTA-family mean + 0.35 * big-launch
mean + 0.15 * full-pool mean, with per-target detail kept visible.

Forecasters:
  F1  the shipped v2 analogue model (tune2.forecast_x, config E)
  F2  top-k local analogue: F1's pipeline on the k heaviest pool members
  F3  smoothed-curve analogue: each launch's log-ratio curve is smoothed
      over hours before quantiles are taken (shape/noise separation)
  F4  global pooled: near-equal weights (family filters only), strong
      normal blend -- a low-variance prior model

Usage: python3 tools/tune3.py [report|blend|calibrate]
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as model  # noqa: E402
import tune2  # noqa: E402

LEVELS = model.LEVELS
Z = model.Z

V2_PARAMS = tune2.merged({
    "w_tier_weak": 0.5, "size_floor": 0.0, "blend_normal": 0.4,
    "spread": 0.7, "sp_g1": 1.15, "sp_g3": 0.85, "sp_g4": 0.85,
    "sp_hi": 0.85, "tail": 1.1, "pre_shrink": 0.5})


def smooth_curve(ratios, width):
    """Centred moving average of a {h: log ratio} curve, edges reflected."""
    if width <= 1:
        return dict(ratios)
    half = width // 2
    hs = sorted(ratios)
    out = {}
    for h in hs:
        vals = []
        for d in range(-half, half + 1):
            hh = h + d
            if hh in ratios:
                vals.append(ratios[hh])
            elif hs:
                vals.append(ratios[hs[0] if hh < hs[0] else hs[-1]])
        out[h] = sum(vals) / len(vals)
    return out


def build_entries(pool, tgt, params, topk=None, smooth_w=1, pre_kernel=0.0):
    entries = []
    for c in pool:
        if c["name"] == tgt["name"]:
            continue
        w = model.comp_weight(c["feat"], {"console_only": True},
                              c["log_h1"], tgt["log_h1"], params)
        if pre_kernel and c["pre3"] is not None and tgt["pre3"] is not None:
            dd = (c["pre3"] - tgt["pre3"]) / pre_kernel
            w *= 0.1 + 0.9 * math.exp(-0.5 * dd * dd)
        if w < params["min_weight"]:
            continue
        ratios = smooth_curve(c["ratios"], smooth_w) if smooth_w > 1 \
            else c["ratios"]
        entries.append((ratios, w, c["log_h1"], c["pre3"], c["vpc"],
                        c["pre_slope"]))
    if topk:
        entries.sort(key=lambda e: -e[1])
        entries = entries[:topk]
    return entries


def lq_rows(forecast):
    """Forecast dict -> {h: [log quantiles]} (unclamped reconstruction)."""
    return {r["hour"]: [math.log(q) for q in r["q"]]
            for r in forecast["hours"]}


def run_forecasters(pool):
    """{target_name: {model_key: {h: [log q]}}} for every LOLO target."""
    out = {}
    for tgt in pool:
        tx = {"pre3": tgt["pre3"], "vpc": tgt["vpc"],
              "pre_slope": tgt["pre_slope"]}
        res = {}
        e1 = build_entries(pool, tgt, V2_PARAMS)
        f1 = tune2.forecast_x(tgt["p1"], e1, V2_PARAMS, tx)
        res["F1"] = lq_rows(f1 if f1 else model.baseline_forecast(tgt["p1"]))
        for k in (6, 10):
            ek = build_entries(pool, tgt, V2_PARAMS, topk=k)
            fk = tune2.forecast_x(tgt["p1"], ek, V2_PARAMS, tx)
            res[f"F2k{k}"] = lq_rows(fk if fk else res["F1"])
        for wdt in (3, 5):
            ew = build_entries(pool, tgt, V2_PARAMS, smooth_w=wdt)
            fw = tune2.forecast_x(tgt["p1"], ew, V2_PARAMS, tx)
            res[f"F3w{wdt}"] = lq_rows(fw if fw else res["F1"])
        p4 = dict(V2_PARAMS)
        p4.update({"size_sigma": 4.0, "size_floor": 0.3,
                   "blend_normal": 0.5, "w_pc": 0.55,
                   "w_simultaneous": 0.25, "pre_shrink": 0.35})
        e4 = build_entries(pool, tgt, p4)
        f4 = tune2.forecast_x(tgt["p1"], e4, p4, tx)
        res["F4"] = lq_rows(f4 if f4 else res["F1"])
        epk = build_entries(pool, tgt, V2_PARAMS, pre_kernel=1.0)
        fpk = tune2.forecast_x(tgt["p1"], epk, V2_PARAMS, tx)
        res["F1pk"] = lq_rows(fpk if fpk else res["F1"])
        out[tgt["name"]] = res
    return out


def pinball_lq(lq, actual):
    total, n = 0.0, 0
    for h in range(2, 25):
        y = actual.get(h)
        if not y or y <= 0 or h not in lq:
            continue
        ly = math.log(max(y, 1.0))
        for tau, lqv in zip(LEVELS, lq[h]):
            d = ly - lqv
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n if n else float("nan")


def blend(fcs, weights):
    keys = list(weights)
    out = {}
    for h in range(2, 25):
        row = []
        for j in range(5):
            row.append(sum(weights[k] * fcs[k][h][j] for k in keys))
        out[h] = row
    return out


def subsets(pool, losses):
    def mean(sel):
        vals = [losses[t["name"]] for t in pool if sel(t)]
        return sum(vals) / len(vals) if vals else float("nan")
    gta = mean(tune2.gta_like)
    big = mean(lambda t: t["p1"] >= 100000)
    full = mean(lambda t: True)
    return {"gta": gta, "big": big, "full": full,
            "obj": gta + 0.35 * big + 0.15 * full}


def report(pool, forecasts):
    keys = ["F1", "F2k6", "F2k10", "F3w3", "F3w5", "F4", "F1pk"]
    for k in keys:
        losses = {t["name"]: pinball_lq(forecasts[t["name"]][k],
                                        t["actual"]) for t in pool}
        s = subsets(pool, losses)
        print(f"{k:6s} GTA {s['gta']:.4f} big {s['big']:.4f} "
              f"full {s['full']:.4f} obj {s['obj']:.4f}")
    return keys


def search_blend(pool, forecasts, keys):
    weights = {k: (1.0 if k == "F1" else 0.0) for k in keys}

    def obj_of(w):
        losses = {}
        for t in pool:
            losses[t["name"]] = pinball_lq(
                blend(forecasts[t["name"]], w), t["actual"])
        return subsets(pool, losses)

    best = obj_of(weights)
    print(f"start (F1 only): obj {best['obj']:.4f} GTA {best['gta']:.4f}")
    for _ in range(60):
        improved = False
        for k in keys:
            for delta in (0.1, 0.2, 0.35):
                if k == "F1":
                    continue
                trial = dict(weights)
                trial[k] = weights[k] + delta
                tot = sum(trial.values())
                trial = {kk: vv / tot for kk, vv in trial.items()}
                s = obj_of(trial)
                if s["obj"] < best["obj"] - 1e-6:
                    best, weights, improved = s, trial, True
                    print(f"  +{k} -> obj {s['obj']:.4f} GTA {s['gta']:.4f} "
                          f"big {s['big']:.4f} full {s['full']:.4f}")
        if not improved:
            break
    print("blend weights:", {k: round(v, 3) for k, v in weights.items()
                             if v > 0.005})
    return weights, best


def calibrate(pool, forecasts, weights):
    """Per-hour spread multipliers on the blend, shrunk coordinate descent.

    A move is accepted only when the robust objective improves by more
    than 0.0008, which keeps the 23 multipliers from fitting noise.
    """
    mult = {h: 1.0 for h in range(2, 25)}

    def apply(t, m):
        base = blend(forecasts[t["name"]], weights)
        out = {}
        lp1 = math.log(t["p1"])
        for h, row in base.items():
            med = row[2] - lp1
            out[h] = [lp1 + med + m[h] * (x - lp1 - med) + 0.0
                      if False else x for x in row]
            out[h] = [row[2] + m[h] * (x - row[2]) for x in row]
        return out

    def obj_of(m):
        losses = {t["name"]: pinball_lq(apply(t, m), t["actual"])
                  for t in pool}
        return subsets(pool, losses)

    best = obj_of(mult)
    print(f"blend obj {best['obj']:.4f}; calibrating per-hour spread...")
    for _ in range(3):
        improved = False
        for h in range(2, 25):
            for v in (0.9, 0.95, 1.05, 1.1):
                if v == mult[h]:
                    continue
                trial = dict(mult)
                trial[h] = mult[h] * v
                s = obj_of(trial)
                if s["obj"] < best["obj"] - 0.0008:
                    best, mult, improved = s, trial, True
                    print(f"  h{h} x{v}: obj {s['obj']:.4f}")
        if not improved:
            break
    print("hour multipliers:", {h: round(mult[h], 2) for h in mult})
    print(f"calibrated: GTA {best['gta']:.4f} big {best['big']:.4f} "
          f"full {best['full']:.4f} obj {best['obj']:.4f}")
    return mult, best


def main():
    case = json.loads((tune2.KIT_INPUT / "p00.in").read_text())
    pool = tune2.load_pool_x(case)
    forecasts = run_forecasters(pool)
    keys = report(pool, forecasts)
    if "blend" in sys.argv or "calibrate" in sys.argv:
        weights, _ = search_blend(pool, forecasts, keys)
        if "calibrate" in sys.argv:
            calibrate(pool, forecasts, weights)


if __name__ == "__main__":
    main()
