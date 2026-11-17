"""Extended LOLO tuner for the GTA VI forecaster (research only).

Builds on tools/lolo.py's universe (the 37 launches inside the practice
cases) and adds, on top of the shipped pipeline in ../main.py:

  * pre-release features per launch, from its own hourly curve before
    day_start: pre3 = mean ln(v_{-k}/v_1) for k=1..3, pre_slope, and
    hour-1 viewers-per-channel (vpc). At runtime the same features come
    from the case's history_hourly / known row, so anything tuned here
    ports into main.py unchanged in spirit.
  * variant knobs: per-hour-block spread multipliers, asymmetric
    lower/upper spread, an extra tail multiplier for the 5%/95%
    quantiles, shrunk regression tilts on pre3 and vpc, and a recency
    weight on pool members.

Objective (robust, not just the 7 GTA-like targets):
    obj = mean(GTA-like) + 0.35 * mean(h1 >= 100k) + 0.15 * mean(all)

Usage:
    python3 tools/tune2.py            # sanity: defaults == shipped model
    python3 tools/tune2.py search     # coordinate descent over the grids
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as model  # noqa: E402

KIT_INPUT = Path(__file__).resolve().parents[2] / \
    "gta6-twitch-ccu-forecast-01-kit" / "input"
LEVELS = model.LEVELS
Z = model.Z


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_pool_x(case):
    """[(launch dict)] with ratios, features, and pre-release features."""
    out = []
    for comp in case["comparables"]:
        feat = model.comp_features(comp)
        if feat["tier"] == "target":
            continue
        hours = model.day_hours(comp)
        v1 = hours.get(1)
        if not v1 or v1 <= 0:
            continue
        ratios = {h: math.log(v / v1) for h, v in hours.items()
                  if h >= 2 and v > 0}
        if len(ratios) < 12:
            continue
        start = parse_ts(comp["day_start"])
        by_idx = {}
        ch1 = None
        for point in comp.get("hourly", []):
            viewers = point.get("viewers")
            if viewers is None:
                continue
            idx = int((parse_ts(point["from"]) - start).total_seconds()
                      // 3600) + 1
            by_idx[idx] = float(viewers)
            if idx == 1:
                ch1 = point.get("channels")
        logv1 = math.log(v1)
        pre = {}
        for k in range(1, 7):
            v = by_idx.get(1 - k)
            if v and v > 0:
                pre[k] = math.log(v) - logv1
        pre3 = (sum(pre[k] for k in (1, 2, 3) if k in pre)
                / max(1, sum(1 for k in (1, 2, 3) if k in pre))) \
            if any(k in pre for k in (1, 2, 3)) else None
        pre6 = (sum(pre[k] for k in (4, 5, 6) if k in pre)
                / max(1, sum(1 for k in (4, 5, 6) if k in pre))) \
            if any(k in pre for k in (4, 5, 6)) else None
        out.append({
            "name": comp["name"], "ratios": ratios, "feat": feat,
            "log_h1": logv1, "actual": hours, "p1": v1,
            "pre3": pre3,
            "pre_slope": (pre3 - pre6) if (pre3 is not None
                                           and pre6 is not None) else None,
            "vpc": (logv1 - math.log(ch1)) if ch1 else None,
            "year": start.year + (start.month - 1) / 12.0,
        })
    return out


# Extra knobs beyond main.PARAMS; defaults reproduce the shipped model.
EXTRA0 = {
    "sp_g1": 1.0, "sp_g2": 1.0, "sp_g3": 1.0, "sp_g4": 1.0,
    "sp_lo": 1.0, "sp_hi": 1.0, "tail": 1.0,
    "pre_shrink": 0.0, "vpc_shrink": 0.0, "slope_shrink": 0.0,
    "recency": 1.0,
}


def _group_mult(h, p):
    if h <= 5:
        return p["sp_g1"]
    if h <= 12:
        return p["sp_g2"]
    if h <= 18:
        return p["sp_g3"]
    return p["sp_g4"]


def forecast_x(p1, entries, params, tx):
    """entries: [(ratios, weight, log_h1, pre3, vpc, pre_slope)]."""
    log_p1 = math.log(p1)
    rows = []
    for h in range(2, 25):
        sample = [(r[h], w, lh1, pre3, vpc, psl) for r, w, lh1, pre3, vpc,
                  psl in entries if h in r]
        if len(sample) < 3:
            rows.append(None)
            continue
        sample.sort(key=lambda t: t[0])
        pairs = [(v, w) for v, w, _, _, _, _ in sample]
        wsum = sum(w for _, w, _, _, _, _ in sample)
        mean = sum(v * w for v, w, _, _, _, _ in sample) / wsum
        var = sum(w * (v - mean) ** 2 for v, w, _, _, _, _ in sample) / wsum
        sd = math.sqrt(max(var, 1e-6))
        centre = mean

        def tilt(idx, shrink, tval):
            nonlocal centre
            if shrink == 0.0 or tval is None:
                return
            pts = [(s[idx], s[0], s[1]) for s in sample if s[idx] is not None]
            if len(pts) < 6:
                return
            tw = sum(w for _, _, w in pts)
            xbar = sum(x * w for x, _, w in pts) / tw
            ybar = sum(y * w for _, y, w in pts) / tw
            sxx = sum(w * (x - xbar) ** 2 for x, _, w in pts)
            sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in pts)
            if sxx > 1e-9:
                centre += shrink * (sxy / sxx) * (tval - xbar)

        # size tilt (same as main.py, shrinkage from params)
        xbar = sum(lh1 * w for _, w, lh1, _, _, _ in sample) / wsum
        sxx = sum(w * (lh1 - xbar) ** 2 for _, w, lh1, _, _, _ in sample)
        sxy = sum(w * (lh1 - xbar) * (v - mean) for v, w, lh1, _, _, _
                  in sample)
        slope = (sxy / sxx) if sxx > 1e-9 else 0.0
        centre += params["tilt_shrink"] * slope * (log_p1 - xbar)
        tilt(3, params["pre_shrink"], tx.get("pre3"))
        tilt(4, params["vpc_shrink"], tx.get("vpc"))
        tilt(5, params["slope_shrink"], tx.get("pre_slope"))
        qs = []
        for tau, z in zip(LEVELS, Z):
            emp = model.weighted_quantile(pairs, tau)
            emp_c = centre + (emp - mean)
            norm = centre + z * sd
            lq = (1.0 - params["blend_normal"]) * emp_c \
                + params["blend_normal"] * norm
            qs.append(lq)
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
    if params.get("smooth"):
        smoothed = []
        for i, qs in enumerate(rows):
            left = rows[i - 1] if i > 0 else qs
            right = rows[i + 1] if i < len(rows) - 1 else qs
            smoothed.append([0.25 * a + 0.5 * b + 0.25 * c
                             for a, b, c in zip(left, qs, right)])
        rows = smoothed
    out = []
    for i, qs in enumerate(rows):
        h = i + 2
        med = qs[2]
        gm = _group_mult(h, params)
        adj = []
        for j, q in enumerate(qs):
            d = (q - med) * params["spread"] * gm
            if j < 2:
                d *= params["sp_lo"]
            elif j > 2:
                d *= params["sp_hi"]
            if j in (0, 4):
                d *= params["tail"]
            adj.append(med + d)
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


def pinball(forecast, actual):
    total, n = 0.0, 0
    rows = {r["hour"]: r["q"] for r in forecast["hours"]}
    for h in range(2, 25):
        y = actual.get(h)
        if not y or y <= 0 or h not in rows:
            continue
        ly = math.log(max(y, 1.0))
        for tau, q in zip(LEVELS, rows[h]):
            d = ly - math.log(max(q, 1e-9))
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n if n else float("nan")


def evaluate_x(pool, params):
    losses = {}
    for i, tgt in enumerate(pool):
        entries = []
        for j, c in enumerate(pool):
            if i == j:
                continue
            w = model.comp_weight(c["feat"], {"console_only": True},
                                  c["log_h1"], tgt["log_h1"], params)
            w *= params["recency"] ** max(0.0, 2026.0 - c["year"])
            if w < params["min_weight"]:
                continue
            entries.append((c["ratios"], w, c["log_h1"], c["pre3"],
                            c["vpc"], c["pre_slope"]))
        fc = forecast_x(tgt["p1"], entries, params,
                        {"pre3": tgt["pre3"], "vpc": tgt["vpc"],
                         "pre_slope": tgt["pre_slope"]})
        if fc is None:
            fc = model.baseline_forecast(tgt["p1"])
        losses[tgt["name"]] = pinball(fc, tgt["actual"])
    return losses


def gta_like(tgt):
    f = tgt["feat"]
    return (f["console_only"] and f["tier"] == "comparable"
            and not f["simultaneous_only"] and not f["early_access"])


def subsets(pool, losses):
    def mean(sel):
        vals = [losses[t["name"]] for t in pool if sel(t)]
        return sum(vals) / len(vals) if vals else float("nan")
    gta = mean(gta_like)
    big = mean(lambda t: t["p1"] >= 100000)
    full = mean(lambda t: True)
    obj = gta + 0.35 * big + 0.15 * full
    return {"gta": gta, "big": big, "full": full, "obj": obj}


def merged(params):
    p = dict(model.PARAMS)
    p.update(EXTRA0)
    p.update(params)
    return p


def main():
    case = json.loads((KIT_INPUT / "p00.in").read_text())
    pool = load_pool_x(case)
    print(f"pool: {len(pool)} launches; pre3 available: "
          f"{sum(1 for t in pool if t['pre3'] is not None)}, "
          f"vpc: {sum(1 for t in pool if t['vpc'] is not None)}")
    params = merged({})
    losses = evaluate_x(pool, params)
    s = subsets(pool, losses)
    print(f"sanity (defaults): GTA {s['gta']:.4f} big {s['big']:.4f} "
          f"full {s['full']:.4f} obj {s['obj']:.4f}")

    if "search" not in sys.argv:
        return
    grid = {
        "w_tier_early": [0.15, 0.25, 0.35, 0.5, 0.7, 1.0],
        "w_tier_weak": [0.05, 0.1, 0.2, 0.3, 0.5],
        "w_pc": [0.2, 0.3, 0.45, 0.55, 0.7, 1.0],
        "w_simultaneous": [0.02, 0.05, 0.15, 0.25, 0.45],
        "size_sigma": [0.7, 0.9, 1.15, 1.5, 2.2, 4.0],
        "size_floor": [0.0, 0.05, 0.15, 0.3],
        "blend_normal": [0.0, 0.15, 0.25, 0.4, 0.6],
        "spread": [0.7, 0.8, 0.9, 1.0, 1.1, 1.25],
        "smooth": [0, 1],
        "tilt_shrink": [0.0, 0.3, 0.5, 0.8, 1.0],
        "min_weight": [0.01, 0.02, 0.05],
        "sp_g1": [0.85, 1.0, 1.15],
        "sp_g2": [0.85, 1.0, 1.15],
        "sp_g3": [0.85, 1.0, 1.15],
        "sp_g4": [0.85, 1.0, 1.15],
        "sp_lo": [0.85, 1.0, 1.15, 1.3],
        "sp_hi": [0.85, 1.0, 1.15, 1.3],
        "tail": [0.9, 1.0, 1.1, 1.25],
        "pre_shrink": [0.0, 0.25, 0.5, 0.75],
        "vpc_shrink": [0.0, 0.25, 0.5],
        "recency": [0.9, 0.95, 1.0],
    }
    best = s["obj"]
    for round_no in range(3):
        improved = False
        for key, values in grid.items():
            for v in values:
                if v == params[key]:
                    continue
                trial = dict(params)
                trial[key] = v
                m = subsets(pool, evaluate_x(pool, trial))["obj"]
                if m < best - 1e-6:
                    best, params = m, trial
                    improved = True
                    print(f"  r{round_no} {key} -> {v}: obj {m:.4f}")
        if not improved:
            break
    losses = evaluate_x(pool, params)
    s = subsets(pool, losses)
    print(f"tuned: GTA {s['gta']:.4f} big {s['big']:.4f} "
          f"full {s['full']:.4f} obj {s['obj']:.4f}")
    print("tuned params:", json.dumps(params, indent=1))
    gta_losses = {t["name"]: round(losses[t["name"]], 3)
                  for t in pool if gta_like(t)}
    print("GTA-like losses:", gta_losses)


if __name__ == "__main__":
    main()
