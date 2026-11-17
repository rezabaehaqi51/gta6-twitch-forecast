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
  single-player target: house tier (early-access ×0.5, live-service ×0.1),
  PC-including ×0.3, single-global-unlock ×0.05, and a Gaussian size
  kernel on `ln(hour-1 viewers)` (sigma 1.5, floor 0.05) — GTA VI's first
  hour will already be hot, so launches that started hot are the evidence.
- Per hour: weighted empirical quantiles of the ratios blended 85/15 with
  a normal fit, centre tilted by a shrunk (×0.5) regression of the ratio
  on launch size, spread calibrated ×0.8 around the median.
- The GTA VI category's own premiere-day record (tier `target`) is never
  used as a launch analogue. Too few analogues → house-baseline fallback.

## Validation (tools/lolo.py)

Leave-one-launch-out over the 37 launches inside the practice cases:
each launch is forecast from the other 36 through the exact production
pipeline and scored with the contest's pinball loss on the log scale.

| target set | this model | house baseline |
| --- | --- | --- |
| GTA-family (console-only, full tier, rolling) | **0.164** | 0.344 |
| four biggest non-early launches (h1 ≥ 250k) | **0.155** | — |
| all 37 launches | 0.478 | 0.930 |

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
