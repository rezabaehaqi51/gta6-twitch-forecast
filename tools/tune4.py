"""Bias-correction + stronger size response for the forecaster.

The v2 model's LOLO residuals show a smooth per-hour median bias for
GTA-family targets (early dip too shallow, mid-day build too weak) and
a broad overforecast of the biggest launches. tune4 tests:

  * a pool-internal LOLO bias correction: every pool member is forecast
    from the other members, its per-hour median residuals are averaged
    (weighted by similarity to the target), smoothed over hours and
    shrunk by kappa, then added to the target's log quantiles. At
    runtime the same computation runs on the case's comparables, so
    nothing about the target leaks in.
  * a stronger / quadratic size tilt.

Usage: python3 tools/tune4.py
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


def entries_for(pool, tgt, params):
    out = []
    for c in pool:
        if c["name"] == tgt["name"]:
            continue
        w = model.comp_weight(c["feat"], {"console_only": True},
                              c["log_h1"], tgt["log_h1"], params)
        if w < params["min_weight"]:
            continue
        out.append((c, (c["ratios"], w, c["log_h1"], c["pre3"], c["vpc"],
                        c["pre_slope"])))
    return out


def base_forecast(pool, tgt, params):
    ents = entries_for(pool, tgt, params)
    fc = tune2.forecast_x(tgt["p1"], [e[1] for e in ents], params,
                          {"pre3": tgt["pre3"], "vpc": tgt["vpc"],
                           "pre_slope": tgt["pre_slope"]})
    if fc is None:
        fc = model.baseline_forecast(tgt["p1"])
    return tune3.lq_rows(fc), ents


def bias_vector(pool, tgt, params, ents):
    """Per-hour weighted mean LOLO median residual inside the pool."""
    resid = {h: [] for h in range(2, 25)}
    members = [c for c, _ in ents]
    wmap = {c["name"]: w for c, (_, w, _, _, _, _) in
            [(c, e) for c, e in ents]}
    for m in members:
        sub = [c for c in members if c["name"] != m["name"]]
        ents_m = []
        for c in sub:
            w = model.comp_weight(c["feat"], {"console_only": True},
                                  c["log_h1"], m["log_h1"], params)
            if w < params["min_weight"]:
                continue
            ents_m.append((c["ratios"], w, c["log_h1"], c["pre3"],
                           c["vpc"], c["pre_slope"]))
        fc = tune2.forecast_x(m["p1"], ents_m, params,
                              {"pre3": m["pre3"], "vpc": m["vpc"],
                               "pre_slope": m["pre_slope"]})
        if fc is None:
            continue
        lq = tune3.lq_rows(fc)
        wm = wmap[m["name"]]
        for h in range(2, 25):
            y = m["actual"].get(h)
            if y and h in lq:
                resid[h].append((math.log(y) - lq[h][2], wm))
    out = {}
    for h, lst in resid.items():
        if lst:
            tw = sum(w for _, w in lst)
            out[h] = sum(r * w for r, w in lst) / tw
    # smooth over neighbouring hours
    sm = {}
    for h in range(2, 25):
        vals = [(out.get(hh), 0.5 if hh == h else 0.25)
                for hh in (h - 1, h, h + 1) if out.get(hh) is not None]
        if vals:
            tw = sum(w for _, w in vals)
            sm[h] = sum(v * w for v, w in vals) / tw
    return sm


def run(pool, params, kappa):
    losses = {}
    for tgt in pool:
        lq, ents = base_forecast(pool, tgt, params)
        if kappa:
            bv = bias_vector(pool, tgt, params, ents)
            for h, row in lq.items():
                if h in bv:
                    lq[h] = [x + kappa * bv[h] for x in row]
        losses[tgt["name"]] = tune3.pinball_lq(lq, tgt["actual"])
    return losses


def main():
    case = json.loads((tune2.KIT_INPUT / "p00.in").read_text())
    pool = tune2.load_pool_x(case)
    base = tune3.V2_PARAMS
    for kappa in (0.0, 0.3, 0.5, 0.7, 1.0):
        s = tune3.subsets(pool, run(pool, base, kappa))
        print(f"kappa {kappa}: GTA {s['gta']:.4f} big {s['big']:.4f} "
              f"full {s['full']:.4f} obj {s['obj']:.4f}")
    # stronger size tilt on top of the best kappa so far
    for tilt in (0.65, 0.8):
        p = dict(base)
        p["tilt_shrink"] = tilt
        s = tune3.subsets(pool, run(pool, p, 0.5))
        print(f"kappa 0.5 + tilt_shrink {tilt}: GTA {s['gta']:.4f} "
              f"big {s['big']:.4f} full {s['full']:.4f} obj {s['obj']:.4f}")


if __name__ == "__main__":
    main()
