# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rezabaehaqi51/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rezabaehaqi51/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An hour-by-hour probabilistic forecast of GTA VI's launch-day Twitch viewership, built as an hour-1 anchored log-ratio analogue model. A launch day's hourly curve is mostly a *shape* — how hour *h* compares with hour 1 — so the program rebuilds that shape at runtime from the ~40 past big launches the case ships: for each comparable it forms the log ratios `ln(v_h / v_1)` over its release day, weights it by analogue quality for a console-only, rolling-midnight, single-player target (house tier, platform mix, unlock pattern, and a Gaussian size kernel on hour-1 viewers), then takes per-hour weighted quantiles of the ratios — empirical quantiles blended 85/15 with a normal fit, the centre tilted by a shrunk regression on launch size, the spread calibrated by leave-one-launch-out validation over the comparables themselves (pinball loss 0.164 on GTA-family targets vs the house baseline's 0.344). Standard library only, no network, no randomness: two runs of a case write the same bytes. See `DESIGN.md` for the full design notes and `tools/lolo.py` for the validation harness.

## Run it

```sh
MAKEFUN_INPUT_DIR=/path/to/case MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=29pd68a1&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
