"""GTA VI launch-day Twitch CCU forecaster — make.fun gta6-twitch-ccu-forecast-01.

Model: hour-1 anchored log-ratio analogue forecasting.

A launch day's hourly curve is mostly a *shape*: how hour h compares with
hour 1. The case ships the full hourly curves of ~40 past big launches
(`comparables`), so the program rebuilds that shape at runtime:

  1. For every comparable launch, form the log ratios r_c(h) = ln(v_h / v_1)
     over its release day (UTC), anchored on its own hour-1 print.
  2. Weight each comparable by how good an analogue it is for the target:
     the house's tier label in its unlock notes (early-access and
     live-service launches count less), console-only vs PC-including,
     rolling-midnight vs single-global-unlock, and a size kernel on
     ln(hour-1 viewers) — a launch whose first hour was already hot behaves
     differently from one that unlocked into an empty category.
  3. Per hour, take the weighted distribution of the log ratios: blend its
     empirical quantiles with a normal fit (weighted mean/sd), tilt the
     centre by shrunk regressions of the ratio on launch size and on
     pre-release heat (how hot the category already was in the three
     hours before the UTC day, relative to hour 1), and widen or narrow
     the spread by calibration factors — global, per hour block,
     asymmetric, and a tail factor — fitted offline by
     leave-one-launch-out validation over the comparables themselves.
  4. Quantiles are hour-1 print x exp(log-ratio quantile), clamped to the
     contest's range and forced non-decreasing.
  5. The shipped forecast is a small ensemble: the pipeline above runs
     four ways — as described; with pool weights sharpened by a
     pre-release-heat kernel (launches whose pre-day heat matches the
     target's count more); with each launch's ratio curve smoothed over
     neighbouring hours before quantiles are taken (shape first, hourly
     noise second); and with a low-variance global parameter set (wide
     size kernel, weak tier discounts) — and the four log-quantile sets
     are averaged with fixed weights (0.60 / 0.18 / 0.12 / 0.10) fitted
     by the same leave-one-launch-out validation. A last calibration
     pass on the blended rows widens the lower side of each hour's
     interval (x1.05 around the median) and narrows the upper side
     (x0.95): across launches the model's misses run asymmetric, with
     downside surprises fatter than upside ones. No single variant beat
     the base model on every validation subset; the blend ties it on
     GTA-family launches and beats it on big launches and on the union
     of the two, which is the posture wanted for a target that is both.

The GTA VI category's own premiere-day entry in `comparables` (tier
"target") is not a game launch and never enters the pool. If a case carries
too few usable comparables, the program falls back to the house baseline's
persistence spread so it still writes a valid forecast.

Standard library only, no network, no randomness: two runs of a case write
the same bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs with -P, which keeps this file's directory off sys.path:
# put it back first so a module shipped beside main.py would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]  # standard normal quantiles

# --- Model parameters (fitted by tools/lolo.py + tools/tune2.py) ---
# LOLO over the 37 launches in the practice cases, scored through the
# shipped ensemble (tools/lolo_ens.py): GTA-family targets score 0.146
# against the house baseline's 0.344 on the same targets; launches with
# hour 1 >= 100k score 0.209; the union of the two sets scores 0.196;
# the full pool scores 0.452. v4 (tools/tune6*.py) changed three things
# from v3: the blend weights, sp_lo, and a post-blend asymmetric
# calibration (POST_LO / POST_HI below). See DESIGN.md.
PARAMS = {
    "w_tier_early": 0.5,       # early-access tier weight factor
    "w_tier_weak": 0.5,        # live-service / weak tier weight factor
    "w_pc": 0.3,               # comparable includes PC (target is console-only)
    "w_simultaneous": 0.05,    # single global unlock, no rolling console wave
    "size_sigma": 1.5,         # size kernel width on ln(hour-1 viewers)
    "size_floor": 0.0,         # size kernel floor
    "blend_normal": 0.4,       # weight of the normal fit vs empirical quantiles
    "spread": 0.7,             # global spread calibration multiplier
    "sp_g1": 1.15,             # spread multiplier, hours 2-5
    "sp_g2": 1.0,              # spread multiplier, hours 6-12
    "sp_g3": 0.85,             # spread multiplier, hours 13-18
    "sp_g4": 0.85,             # spread multiplier, hours 19-24
    "sp_lo": 1.05,             # extra multiplier below the median
    "sp_hi": 0.85,             # extra multiplier above the median
    "tail": 1.1,               # extra multiplier on the 5%/95% quantiles
    "smooth": 0,               # 1: smooth log quantiles over neighbouring hours
    "tilt_shrink": 0.5,        # shrinkage of the size-tilt slope
    "pre_shrink": 0.5,         # shrinkage of the pre-release heat tilt
    "min_weight": 0.05,        # drop comparables weighted below this
    "min_pool": 6,             # fall back when fewer usable comparables
}

BASELINE_SIGMA = 0.15  # house baseline persistence spread (fallback only)

# The ensemble's global member: same pipeline, low-variance parameters
# (wide size kernel with a floor, gentler tier discounts, stronger
# normal blend, weaker heat tilt). See tools/tune3.py / tune5.py.
GLOBAL_PARAMS = dict(PARAMS, size_sigma=4.0, size_floor=0.3,
                     blend_normal=0.5, w_pc=0.55, w_simultaneous=0.25,
                     pre_shrink=0.35)

# Ensemble weights over (base, heat-kernel, smoothed-curves, global),
# fitted by leave-one-launch-out on the union of GTA-family and big
# launches (tools/tune5.py, re-weighted by tools/tune6.py for v4):
# union 0.196 vs the base model's 0.205, GTA-family 0.146,
# big launches 0.209.
BLEND = (0.60, 0.18, 0.12, 0.10)
# Post-blend asymmetric calibration around each hour's blended median:
# the lower side widens, the upper side narrows (tools/tune6.py).
# LOLO union coverage of the 90% interval rises 0.820 -> 0.852.
POST_LO = 1.05
POST_HI = 0.95
PRE_KERNEL_SIGMA = 1.0   # width of the pre-release heat kernel
SMOOTH_WIDTH = 5         # moving-average width for the smoothed member


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_hours(comp):
    """Hour index (1..24 of the release day) -> viewers, from a comparable."""
    start = parse_ts(comp["day_start"])
    out = {}
    for point in comp.get("hourly", []):
        viewers = point.get("viewers")
        if viewers is None:
            continue
        t = parse_ts(point["from"])
        h = int((t - start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = float(viewers)
    return out


_TIER_RE = re.compile(r"Tier:\s*([A-Za-z]+)")
_EARLY_ACCESS_RE = re.compile(r"early access", re.IGNORECASE)
_NO_EARLY_RE = re.compile(r"(no|without|not)\b[^.]{0,40}early access", re.IGNORECASE)


def comp_features(comp):
    """Analogue features of one comparable launch, from its case record."""
    unlock = comp.get("unlock") or ""
    m = _TIER_RE.search(unlock)
    tier = m.group(1).lower() if m else "comparable"
    # `platforms` arrives as one comma-separated string ("PC, PS5, Xbox
    # Series"), in some records as a list; normalise before reading it.
    platforms = comp.get("platforms") or ""
    if isinstance(platforms, list):
        platforms = ", ".join(str(p) for p in platforms)
    has_pc = bool(re.search(r"\bPC\b", platforms)) or "Stadia" in platforms
    low = unlock.lower()
    simultaneous_only = ("single global unlock" in low and "rolling" not in low) or \
        ("simultaneous" in low and "rolling" not in low and "midnight local" not in low)
    early_access = tier == "early" or (
        bool(_EARLY_ACCESS_RE.search(unlock)) and not _NO_EARLY_RE.search(unlock))
    return {
        "tier": tier,
        "console_only": not has_pc,
        "simultaneous_only": simultaneous_only,
        "early_access": early_access,
    }


def target_profile(case):
    """The target launch's profile, parsed from the case's own notes.

    The final's target is GTA VI: PS5 + Xbox Series, rolling midnight
    local, single-player, no early access. The practice targets are all
    console rolling-midnight launches too; parsing the notes keeps the
    program honest on both instead of hardcoding one game.
    """
    notes = (case.get("notes") or "")
    low = notes.lower()
    console_only = ("pc" not in low.split("released")[0][-200:]) if "released" in low else True
    # Simpler robust read: a target that ships on PC says so in its notes.
    ships_pc = bool(re.search(r"\bon pc\b|\bpc\b[^.]{0,30}\b(launch|release|version)", low))
    early_access = bool(_EARLY_ACCESS_RE.search(notes)) and not _NO_EARLY_RE.search(notes)
    return {"console_only": not ships_pc, "early_access": early_access}


def comp_weight(feat, target, log_h1, log_p1, params):
    w = 1.0
    if feat["tier"] == "early" or feat["early_access"]:
        w *= params["w_tier_early"]
    if feat["tier"] == "weak":
        w *= params["w_tier_weak"]
    if target["console_only"] and not feat["console_only"]:
        w *= params["w_pc"]
    if feat["simultaneous_only"]:
        w *= params["w_simultaneous"]
    d = (log_h1 - log_p1) / params["size_sigma"]
    w *= params["size_floor"] + (1.0 - params["size_floor"]) * math.exp(-0.5 * d * d)
    return w


def weighted_quantile(sorted_pairs, tau):
    """Weighted quantile of [(value, weight)] sorted by value, interpolated."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    threshold = tau * total
    acc = 0.0
    prev_v, prev_acc = sorted_pairs[0][0], 0.0
    for v, w in sorted_pairs:
        acc += w
        if acc >= threshold:
            if acc == prev_acc:
                return v
            frac = (threshold - prev_acc) / (acc - prev_acc)
            return prev_v + frac * (v - prev_v)
        prev_v, prev_acc = v, acc
    return sorted_pairs[-1][0]


def pre_release_heat(points, day_start, log_ref):
    """Mean ln(viewers / exp(log_ref)) over the 3 hours before day_start.

    `points` are hourly rows with `from`/`viewers` (a comparable's
    `hourly`, or the case's `history_hourly` for the target itself).
    Returns None when none of those three hours is present. A launch
    whose category was already hot in the hours before the UTC day
    tends to run a flatter day than one that starts cold, so this is a
    centre-tilt feature, fitted like the size tilt.
    """
    start = parse_ts(day_start)
    vals = []
    for point in points or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        idx = int((parse_ts(point["from"]) - start).total_seconds()
                  // 3600) + 1
        if idx in (0, -1, -2):
            vals.append(math.log(float(viewers)) - log_ref)
    return sum(vals) / len(vals) if vals else None


def build_pool(case, params):
    """Usable comparables: [(ratios, weight, log_h1, pre3)]."""
    known = next((k for k in case["known"] if k["hour"] == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    log_p1 = math.log(p1)
    target = target_profile(case)
    game_name = (case.get("game") or {}).get("name", "")
    pool = []
    for comp in case.get("comparables", []):
        feat = comp_features(comp)
        if feat["tier"] == "target" or comp.get("name") == game_name:
            continue  # the target's own premiere entry is not a launch
        hours = day_hours(comp)
        v1 = hours.get(1)
        if not v1 or v1 <= 0:
            continue
        ratios = {h: math.log(v / v1) for h, v in hours.items()
                  if h >= 2 and v > 0}
        if len(ratios) < 12:
            continue
        w = comp_weight(feat, target, math.log(v1), log_p1, params)
        if w < params["min_weight"]:
            continue
        pre3 = pre_release_heat(comp.get("hourly"), comp["day_start"],
                                math.log(v1))
        pool.append((ratios, w, math.log(v1), pre3))
    return p1, pool


def _group_mult(h, params):
    if h <= 5:
        return params["sp_g1"]
    if h <= 12:
        return params["sp_g2"]
    if h <= 18:
        return params["sp_g3"]
    return params["sp_g4"]


def smooth_curve(ratios, width):
    """Centred moving average of a {hour: log ratio} curve.

    Neighbours outside the recorded hours take the nearest recorded
    value, so the curve's ends flatten instead of dropping out.
    """
    if width <= 1:
        return dict(ratios)
    half = width // 2
    keys = sorted(ratios)
    out = {}
    for h in keys:
        vals = []
        for d in range(-half, half + 1):
            hh = h + d
            if hh in ratios:
                vals.append(ratios[hh])
            else:
                vals.append(ratios[keys[0] if hh < keys[0] else keys[-1]])
        out[h] = sum(vals) / len(vals)
    return out


def log_rows_from_pool(p1, pool, params, target_pre3=None):
    """The 23 rows of calibrated log-ratio quantiles from a built pool.

    Pool entries are (ratios, weight, log_h1, pre3); `target_pre3` is the
    target's own pre-release heat (see pre_release_heat), or None.
    Returns None when the pool yields no usable hour at all.
    """
    log_p1 = math.log(p1)
    # Per-hour weighted centre, spread and tilts from the pool.
    rows = []
    for h in range(2, 25):
        sample = [(r[h], w, lh1, pre3) for r, w, lh1, pre3 in pool
                  if h in r]
        if len(sample) < 3:
            rows.append(None)
            continue
        sample.sort(key=lambda t: t[0])
        pairs = [(v, w) for v, w, _, _ in sample]
        wsum = sum(w for _, w, _, _ in sample)
        mean = sum(v * w for v, w, _, _ in sample) / wsum
        var = sum(w * (v - mean) ** 2 for v, w, _, _ in sample) / wsum
        sd = math.sqrt(max(var, 1e-6))
        # Size tilt: does the log ratio lean on launch size? Weighted
        # regression slope of ratio on ln(hour-1), shrunk toward zero.
        xbar = sum(lh1 * w for _, w, lh1, _ in sample) / wsum
        sxx = sum(w * (lh1 - xbar) ** 2 for _, w, lh1, _ in sample)
        sxy = sum(w * (lh1 - xbar) * (v - mean) for v, w, lh1, _ in sample)
        slope = (sxy / sxx) if sxx > 1e-9 else 0.0
        centre = mean + params["tilt_shrink"] * slope * (log_p1 - xbar)
        # Pre-release heat tilt: the same shrunk regression on the
        # pool's pre-release heat, applied when the target's is known.
        if params["pre_shrink"] and target_pre3 is not None:
            pts = [(pre3, v, w) for v, w, _, pre3 in sample
                   if pre3 is not None]
            if len(pts) >= 6:
                tw = sum(w for _, _, w in pts)
                pxbar = sum(x * w for x, _, w in pts) / tw
                pybar = sum(y * w for _, y, w in pts) / tw
                psxx = sum(w * (x - pxbar) ** 2 for x, _, w in pts)
                psxy = sum(w * (x - pxbar) * (y - pybar)
                           for x, y, w in pts)
                if psxx > 1e-9:
                    centre += params["pre_shrink"] * (psxy / psxx) \
                        * (target_pre3 - pxbar)
        qs = []
        for tau, z in zip(LEVELS, Z):
            emp = weighted_quantile(pairs, tau)
            emp_c = centre + (emp - mean)  # empirical shape, tilted centre
            norm = centre + z * sd
            lq = (1.0 - params["blend_normal"]) * emp_c + params["blend_normal"] * norm
            qs.append(lq)
        rows.append(qs)
    # Fill any missing hour from its neighbours (or the day's mean shape).
    known_rows = [r for r in rows if r is not None]
    if not known_rows:
        return None
    mean_row = [sum(r[i] for r in known_rows) / len(known_rows) for i in range(5)]
    for i, r in enumerate(rows):
        if r is None:
            left = rows[i - 1] if i > 0 else None
            right = next((rows[j] for j in range(i + 1, 23) if rows[j] is not None), None)
            if left is not None and right is not None:
                rows[i] = [(a + b) / 2.0 for a, b in zip(left, right)]
            elif left is not None:
                rows[i] = list(left)
            elif right is not None:
                rows[i] = list(right)
            else:
                rows[i] = list(mean_row)
    if params.get("smooth"):
        # Launch-day shapes are smooth in h; per-hour quantiles from a few
        # dozen launches are not. A light [0.25, 0.5, 0.25] kernel over
        # neighbouring hours trades a little shape detail for less noise.
        smoothed = []
        for i, qs in enumerate(rows):
            left = rows[i - 1] if i > 0 else qs
            right = rows[i + 1] if i < len(rows) - 1 else qs
            smoothed.append([0.25 * a + 0.5 * b + 0.25 * c
                             for a, b, c in zip(left, qs, right)])
        rows = smoothed
    # Spread calibration around each hour's median (the tau=0.5 entry):
    # a global factor, a per-hour-block factor, an asymmetric factor
    # below/above the median, and an extra factor on the tail quantiles.
    calibrated = []
    for i, qs in enumerate(rows):
        gm = _group_mult(i + 2, params)
        med = qs[2]
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
        calibrated.append(adj)
    return calibrated


def format_rows(p1, rows):
    """Log-ratio quantile rows -> the contest's viewer-quantile rows."""
    out = []
    for i, qs in enumerate(rows):
        vals = []
        prev = 0.0
        for lq in qs:
            v = p1 * math.exp(lq)
            v = min(max(v, 1.0), 1_000_000_000.0)
            v = max(v, prev)  # quantiles never decrease
            vals.append(round(v, 2))
            prev = v
        out.append({"hour": i + 2, "q": vals})
    return {"hours": out}


def forecast_from_pool(p1, pool, params, target_pre3=None):
    """The 23 forecast rows from a built pool (shared by entry + validator)."""
    rows = log_rows_from_pool(p1, pool, params, target_pre3)
    if rows is None:
        return None
    return format_rows(p1, rows)


def ensemble_rows(p1, pool, global_pool, target_pre3):
    """Blend the four ensemble members' log-quantile rows.

    Members: the base pool as built; the base pool with weights
    sharpened by the pre-release heat kernel; the base pool with each
    launch's curve smoothed over neighbouring hours; and the pool
    rebuilt under the global parameters. A member that yields nothing
    drops out and the remaining weights renormalise.
    """
    members = []
    base = log_rows_from_pool(p1, pool, PARAMS, target_pre3)
    if base is not None:
        members.append((BLEND[0], base))
    if target_pre3 is not None:
        kern = []
        for ratios, w, lh1, pre3 in pool:
            if pre3 is not None:
                d = (pre3 - target_pre3) / PRE_KERNEL_SIGMA
                w = w * (0.1 + 0.9 * math.exp(-0.5 * d * d))
            if w >= PARAMS["min_weight"]:
                kern.append((ratios, w, lh1, pre3))
        if len(kern) >= 3:
            rows = log_rows_from_pool(p1, kern, PARAMS, target_pre3)
            if rows is not None:
                members.append((BLEND[1], rows))
    smoothed = [(smooth_curve(r, SMOOTH_WIDTH), w, lh1, pre3)
                for r, w, lh1, pre3 in pool]
    rows = log_rows_from_pool(p1, smoothed, PARAMS, target_pre3)
    if rows is not None:
        members.append((BLEND[2], rows))
    if global_pool is not None and len(global_pool) >= 3:
        rows = log_rows_from_pool(p1, global_pool, GLOBAL_PARAMS,
                                  target_pre3)
        if rows is not None:
            members.append((BLEND[3], rows))
    if not members:
        return None
    total = sum(w for w, _ in members)
    blended = []
    for i in range(23):
        blended.append([sum(w * rows[i][j] for w, rows in members) / total
                        for j in range(5)])
    # Post-blend asymmetric calibration: widen below the blended
    # median, narrow above it (POST_LO / POST_HI).
    calibrated = []
    for qs in blended:
        med = qs[2]
        calibrated.append([med + (q - med) * (POST_LO if j < 2 else
                                              POST_HI if j > 2 else 1.0)
                           for j, q in enumerate(qs)])
    return calibrated


def baseline_forecast(p1):
    """The house baseline's persistence forecast (fallback only)."""
    out = []
    for h in range(2, 25):
        vals = []
        prev = 0.0
        for z in Z:
            v = p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))
            v = min(max(round(v, 2), 1.0), 1_000_000_000.0)
            v = max(v, prev)
            vals.append(v)
            prev = v
        out.append({"hour": h, "q": vals})
    return {"hours": out}


def forecast_case(case, params=PARAMS):
    if case["levels"] != LEVELS:
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    p1, pool = build_pool(case, params)
    if not pool:
        return baseline_forecast(p1)
    target_pre3 = pre_release_heat(case.get("history_hourly"),
                                   case["day_start"], math.log(p1))
    _, global_pool = build_pool(case, GLOBAL_PARAMS)
    rows = ensemble_rows(p1, pool, global_pool, target_pre3)
    if rows is None:
        return baseline_forecast(p1)
    return format_rows(p1, rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
