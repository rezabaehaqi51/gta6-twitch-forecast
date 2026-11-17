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
     centre by a shrunk regression of the ratio on launch size, and widen
     or narrow the spread by a calibration factor fitted offline by
     leave-one-launch-out validation over the comparables themselves.
  4. Quantiles are hour-1 print x exp(log-ratio quantile), clamped to the
     contest's range and forced non-decreasing.

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

# --- Model parameters (fitted by tools/lolo.py, leave-one-launch-out) ---
# LOLO over the 37 launches in the practice cases: GTA-family targets
# (console-only, full tier, rolling unlock) score 0.164 against the house
# baseline's 0.344 on the same targets; the four biggest non-early
# launches score 0.155. See tools/lolo.py and DESIGN.md.
PARAMS = {
    "w_tier_early": 0.5,       # early-access tier weight factor
    "w_tier_weak": 0.1,        # live-service / weak tier weight factor
    "w_pc": 0.3,               # comparable includes PC (target is console-only)
    "w_simultaneous": 0.05,    # single global unlock, no rolling console wave
    "size_sigma": 1.5,         # size kernel width on ln(hour-1 viewers)
    "size_floor": 0.05,        # size kernel floor
    "blend_normal": 0.15,      # weight of the normal fit vs empirical quantiles
    "spread": 0.8,             # global spread calibration multiplier
    "smooth": 0,               # 1: smooth log quantiles over neighbouring hours
    "tilt_shrink": 0.5,        # shrinkage of the size-tilt slope
    "min_weight": 0.05,        # drop comparables weighted below this
    "min_pool": 6,             # fall back when fewer usable comparables
}

BASELINE_SIGMA = 0.15  # house baseline persistence spread (fallback only)


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


def build_pool(case, params):
    """Usable comparables: [(ratios {h: log ratio}, weight, log_h1)]."""
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
        pool.append((ratios, w, math.log(v1)))
    return p1, pool


def forecast_from_pool(p1, pool, params):
    """The 23 forecast rows from a built pool (shared by entry + validator)."""
    log_p1 = math.log(p1)
    # Per-hour weighted centre, spread and size tilt from the pool.
    rows = []
    for h in range(2, 25):
        sample = [(r[h], w, lh1) for r, w, lh1 in pool if h in r]
        if len(sample) < 3:
            rows.append(None)
            continue
        sample.sort(key=lambda t: t[0])
        pairs = [(v, w) for v, w, _ in sample]
        wsum = sum(w for _, w, _ in sample)
        mean = sum(v * w for v, w, _ in sample) / wsum
        var = sum(w * (v - mean) ** 2 for v, w, _ in sample) / wsum
        sd = math.sqrt(max(var, 1e-6))
        # Size tilt: does the log ratio lean on launch size? Weighted
        # regression slope of ratio on ln(hour-1), shrunk toward zero.
        xbar = sum(lh1 * w for _, w, lh1 in sample) / wsum
        sxx = sum(w * (lh1 - xbar) ** 2 for _, w, lh1 in sample)
        sxy = sum(w * (lh1 - xbar) * (v - mean) for v, w, lh1 in sample)
        slope = (sxy / sxx) if sxx > 1e-9 else 0.0
        centre = mean + params["tilt_shrink"] * slope * (log_p1 - xbar)
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
    # Spread calibration around each hour's median (the tau=0.5 entry).
    rows = [[qs[2] + params["spread"] * (q - qs[2]) for q in qs] for qs in rows]
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
    if len(pool) < params["min_pool"]:
        return baseline_forecast(p1)
    result = forecast_from_pool(p1, pool, params)
    if result is None:
        return baseline_forecast(p1)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
