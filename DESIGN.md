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

## Data notes

- `platforms` in a case is one comma-separated string, not a list — the
  first version of the feature parser iterated it character by character
  and silently marked every launch console-only. LOLO caught it: the PC
  weight had no effect. Fixed in `comp_features`.
- SullyGnome hourly averages are the practice/comparable source (kit
  NOTICE.txt); the final's prints are the house's own minute reads.
