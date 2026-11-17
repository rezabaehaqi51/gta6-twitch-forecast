"""Leave-one-launch-out validation for the forecaster in ../main.py.

Every comparable launch in a practice case is, in turn, treated as the
target: it is forecast from the other launches with exactly the pipeline
main.py runs (same weights, same quantile construction), then scored with
the contest's pinball loss on the log scale against its real hours 2-24.
That is an honest out-of-sample estimate: no launch ever sees itself.

Usage:
    python3 tools/lolo.py                 # score the shipped PARAMS + baseline
    python3 tools/lolo.py --tune          # coordinate-descent grid search
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as model  # noqa: E402

KIT_INPUT = Path(__file__).resolve().parents[2] / \
    "gta6-twitch-ccu-forecast-01-kit" / "input"
LEVELS = model.LEVELS


def load_pool(case):
    """[(name, ratios, features, log_h1, actual {h: viewers})] per comparable."""
    out = []
    for comp in case["comparables"]:
        feat = model.comp_features(comp)
        if feat["tier"] == "target":
            continue
        hours = model.day_hours(comp)
        v1 = hours.get(1)
        if not v1 or v1 <= 0:
            continue
        ratios = {h: math.log(v / v1) for h, v in hours.items() if h >= 2 and v > 0}
        if len(ratios) < 12:
            continue
        out.append({
            "name": comp["name"], "ratios": ratios, "feat": feat,
            "log_h1": math.log(v1), "actual": hours, "p1": v1,
            "pre3": model.pre_release_heat(comp.get("hourly"),
                                           comp["day_start"],
                                           math.log(v1)),
        })
    return out


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


def evaluate(pool, params):
    losses = {}
    for i, tgt in enumerate(pool):
        sub = []
        for j, c in enumerate(pool):
            if i == j:
                continue
            w = model.comp_weight(c["feat"], {"console_only": True},
                                  c["log_h1"], tgt["log_h1"], params)
            if w < params["min_weight"]:
                continue
            sub.append((c["ratios"], w, c["log_h1"], c["pre3"]))
        fc = model.forecast_from_pool(tgt["p1"], sub, params, tgt["pre3"])
        if fc is None:
            fc = model.baseline_forecast(tgt["p1"])
        losses[tgt["name"]] = pinball(fc, tgt["actual"])
    return losses


def gta_like(tgt):
    """Targets in GTA VI's family: console-only, full-price tier, rolling."""
    f = tgt["feat"]
    return (f["console_only"] and f["tier"] == "comparable"
            and not f["simultaneous_only"] and not f["early_access"])


def subset_mean(pool, losses):
    sub = [losses[t["name"]] for t in pool if gta_like(t)]
    return sum(sub) / len(sub) if sub else float("nan")


def main():
    case = json.loads((KIT_INPUT / "p00.in").read_text())
    pool = load_pool(case)
    n_like = sum(1 for t in pool if gta_like(t))
    print(f"pool: {len(pool)} launches ({n_like} GTA-like)")
    base_losses = {}
    for tgt in pool:
        base_losses[tgt["name"]] = pinball(
            model.baseline_forecast(tgt["p1"]), tgt["actual"])
    base = sum(base_losses.values()) / len(base_losses)
    print(f"house baseline (persistence) LOLO loss: {base:.4f} "
          f"(GTA-like {subset_mean(pool, base_losses):.4f})")

    params = dict(model.PARAMS)
    losses = evaluate(pool, params)
    mean = sum(losses.values()) / len(losses)
    print(f"model (shipped params) LOLO loss:      {mean:.4f} "
          f"(GTA-like {subset_mean(pool, losses):.4f})")

    if "--tune" in sys.argv:
        # Optimise the GTA-like subset: the final's target (and every
        # practice target) sits in that family, so the full-pool mean --
        # dominated by near-empty-hour-1 global unlocks -- is the wrong
        # objective. The full-pool loss is still printed as a sanity check.
        grid = {
            "w_tier_early": [0.15, 0.25, 0.35, 0.5, 0.7, 1.0],
            "w_tier_weak": [0.1, 0.2, 0.3, 0.5, 1.0],
            "w_pc": [0.3, 0.45, 0.55, 0.7, 1.0],
            "w_simultaneous": [0.05, 0.15, 0.25, 0.45, 1.0],
            "size_sigma": [0.7, 0.9, 1.15, 1.5, 2.2, 4.0],
            "size_floor": [0.0, 0.05, 0.15],
            "blend_normal": [0.0, 0.15, 0.25, 0.4, 0.6],
            "spread": [0.8, 0.9, 1.0, 1.1, 1.25, 1.45],
            "smooth": [0, 1],
            "tilt_shrink": [0.0, 0.3, 0.5, 0.8, 1.0],
            "min_weight": [0.01, 0.02, 0.05],
        }
        best = subset_mean(pool, losses)
        for round_no in range(3):
            improved = False
            for key, values in grid.items():
                for v in values:
                    if v == params[key]:
                        continue
                    trial = dict(params)
                    trial[key] = v
                    m = subset_mean(pool, evaluate(pool, trial))
                    if m < best - 1e-6:
                        best, params = m, trial
                        improved = True
                        print(f"  round {round_no}: {key} -> {v}: "
                              f"GTA-like {m:.4f}")
            if not improved:
                break
        losses = evaluate(pool, params)
        full = sum(losses.values()) / len(losses)
        print(f"tuned: GTA-like {best:.4f}, full pool {full:.4f}")
        print("tuned params:", json.dumps(params, indent=1))

    worst = sorted(losses.items(), key=lambda kv: -kv[1])[:6]
    print("worst launches:", [(n, round(v, 3)) for n, v in worst])
    best5 = sorted(losses.items(), key=lambda kv: kv[1])[:5]
    print("best launches: ", [(n, round(v, 3)) for n, v in best5])


if __name__ == "__main__":
    main()
