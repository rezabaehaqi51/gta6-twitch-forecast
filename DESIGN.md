# Design notes — GTA VI launch-day Twitch forecast

## The bet

The final is a one-shot forecast nobody can replay: the case is cut at the
end of hour 1 on 2026-11-19 and the target day has never happened. The
practice board's perfect scores come from the practice case carrying its
own answer's neighbourhood; the final rewards the best calibrated
probabilistic model of a launch day, so that is what this entry is.

## Model (main.py)

Hour-1 anchored log-ratio analogues, rebuilt from the case at runtime:

- For each comparable launch: `r_c(h) = ln(v_h / v_1)` over its release day.
- Weights = analogue quality for a console-only, rolling-midnight,
  single-player target: house tier (early-access ×0.5, live-service ×0.5),
  PC-including ×0.3, single-global-unlock ×0.05, and a Gaussian size
  kernel on `ln(hour-1 viewers)` (sigma 1.5, floor 0) — GTA VI's first
  hour will already be hot, so launches that started hot are the evidence.
- Per hour: weighted empirical quantiles of the ratios blended 60/40 with
  a normal fit, centre tilted by shrunk (×0.5) regressions of the ratio
  on launch size and on pre-release heat (the mean log level of the
  three hours before the UTC day relative to hour 1: a category already
  hot before midnight runs a flatter day), spread calibrated around the
  median (×0.7 global; ×1.15 for hours 2–5, ×0.85 for hours 13–24 and
  above the median, ×1.1 on the tail quantiles).
- The GTA VI category's own premiere-day record (tier `target`) is never
  used as a launch analogue. Too few analogues → house-baseline fallback.

## Validation (tools/lolo.py)

Leave-one-launch-out over the 37 launches inside the practice cases:
each launch is forecast from the other 36 through the exact production
pipeline and scored with the contest's pinball loss on the log scale.

| target set | this model (v2) | v1 | house baseline |
| --- | --- | --- | --- |
| GTA-family (console-only, full tier, rolling) | **0.144** | 0.164 | 0.344 |
| launches with hour 1 ≥ 100k (16) | **0.220** | 0.230 | — |
| all 37 launches | 0.450 | 0.478 | 0.930 |

v2 (tools/tune2.py) was tuned on a robust objective — GTA-family mean
plus the big-launch and full-pool means — not on the 7 GTA-family
targets alone: it wins 24 of 37 targets against v1, its GTA-family
jackknife range (0.131–0.153) sits entirely below v1's (0.156–0.173),
and the single biggest gain is the pre-release heat tilt. Knobs that
only fit noise (a recency weight, viewers-per-channel and pre-release
slope tilts, finer second-round spread moves) were tested and dropped.

The all-launches number is worse by design: the model specialises away
from near-empty-hour-1 global unlocks (Wukong, Silksong), whose day shape
says nothing about GTA VI's. The practice targets (Wolverine, RDR2, God
of War Ragnarök, Spider-Man 2) all sit in the GTA family.

## v3 — an ensemble on top of v2 (tools/tune3.py, tune4.py, tune5.py)

GTA VI is GTA-family by construction and almost surely big (its own
category printed 1.42M viewers for a premiere event), so v3 is judged
on the union of the two validation subsets. Around the v2 pipeline,
four members were built and blended in log-quantile space: the base
model; the base with pool weights sharpened by a pre-release-heat
kernel; the base over hour-smoothed ratio curves; and a low-variance
global parameter set. The shipped weights (0.65 / 0.18 / 0.12 / 0.05,
base / heat-kernel / smoothed / global) score:

| target set | v3 ensemble | v2 |
| --- | --- | --- |
| GTA-family | 0.144 | 0.144 |
| launches with hour 1 ≥ 100k | **0.214** | 0.220 |
| union of the two (21) | **0.200** | 0.205 |
| all 37 launches | 0.449 | 0.450 |

Tested and rejected on the same validation: per-hour spread
calibration on top of the blend (nothing survived the acceptance
threshold), a pool-internal LOLO bias correction (helped the full
pool, hurt the GTA family), a family-only bias correction and a
family-specialist blend (same trade), peak-aligned curve averaging
(worse everywhere), top-k local pools, a stronger size tilt, and a
cap on the size tilt's extrapolation. The shipped program reproduces
the blend's LOLO numbers exactly; one divergence from the tuner was
found and fixed in the process — a minimum-pool gate in the entry
path sent small-hour-1 targets to the persistence fallback that the
validation never used.

## v4 — re-weighted blend + asymmetric post-calibration (tools/tune6*.py)

v4 was tuned against the shipped ensemble itself, through a new
harness (tools/lolo_ens.py) that rebuilds each target's base and
global sub-pools and calls main.ensemble_rows — it reproduces v3's
numbers exactly before any variant is trusted (union 0.1996,
GTA-family 0.144). 105 variants were scored: blend-weight moves,
post-blend spread calibration (global / per-block / asymmetric /
tail), every PARAMS and GLOBAL_PARAMS knob one at a time, the heat
kernel's width and the smoothing width. Diagnostics
(tools/diag_spread.py) showed plain widening buys its union gain on
five mid-size off-family targets whose shapes the pool misses, while
taxing the GTA family — so the adopted stack takes the widening only
where the residuals say it belongs:

* blend weights (0.65 / 0.18 / 0.12 / 0.05) -> (0.60 / 0.18 / 0.12 /
  0.10): the low-variance global member earns a larger share;
* sp_lo 1.0 -> 1.05 in the members' own calibration (the global
  member inherits it);
* a post-blend asymmetric calibration on the blended rows: below
  each hour's median x1.05, above it x0.95 (POST_LO / POST_HI).

| target set | v4 | v3 |
| --- | --- | --- |
| GTA-family | 0.146 | 0.144 |
| launches with hour 1 ≥ 100k | **0.209** | 0.214 |
| union of the two (21) | **0.196** | 0.200 |
| all 37 launches | 0.452 | 0.449 |
| union coverage of the 90% interval | **0.857** | 0.820 |

Robustness: the candidate wins both split halves of the union set,
wins all 20 leave-one-target-out jackknife drops, and beats v3 on
12 of 21 union targets head-to-head. Rejected at this gate: a wider
global spread (union 0.195 at spread 0.75, but GTA-family 0.148 —
spends nearly the whole family budget for a worse trade), per-hour
spread calibration on the blend, tilt/spread knob moves that only
re-shuffle the same trade, and seven probes around the adopted stack
that gained <= 0.0005 (inside the noise band). The full-pool mean
worsens slightly (0.449 -> 0.452), the same deliberate specialisation
away from off-posture targets as v2/v3.

## Data notes

- `platforms` in a case is one comma-separated string, not a list — the
  first version of the feature parser iterated it character by character
  and silently marked every launch console-only. LOLO caught it: the PC
  weight had no effect. Fixed in `comp_features`.
- SullyGnome hourly averages are the practice/comparable source (kit
  NOTICE.txt); the final's prints are the house's own minute reads.
