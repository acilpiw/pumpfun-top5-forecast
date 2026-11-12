# pump.fun top-5 forecaster

Forecasts which five pump.fun coins launched since the opening will be the
largest by market cap at each of the house's seven answer reads, and how
large each place's market cap will be.

## Method

Monte Carlo log-cap diffusion, standard library only:

- Every coin's log10 market cap (in SOL) is tracked across the 10-minute
  snapshots. A drift (log10 per hour) and a volatility are fitted per coin
  by least squares and shrunk toward the population median, with the drift
  rate capped: a coin cannot sustain a multi-x-per-hour climb.
- Each coin's log-cap is extrapolated to every answer read in `task.json`
  (the drift displacement is capped, and the sampling spread grows with the
  horizon: a base term, the coin's own diffusion, and a term that grows
  linearly with horizon for drift error and jumps, capped overall). The
  spread grows faster than sqrt(dt) because the backtest showed the future
  is far more dispersed than a plain diffusion implies: the top 5 turns
  over almost completely within an hour.
- 2,500 fixed-seed Monte Carlo draws per coin are ranked at each read.
  Place probabilities are the draw frequencies averaged over the reads
  (the one forecast is scored at every read, so the mixture is the honest
  single forecast), scaled down by the share of top-5 slots a coin born
  after the cut is expected to take. Those slots are scored from the
  unnamed mass, so the forecast leaves that mass there instead of spending
  it on coins that cannot hold it.
- Each place's size distribution is the Monte Carlo histogram of the
  log-cap at that place, widened with a broad normal to cover newcomers
  and model error.

## Files

- `main.py` — the entry. Reads `$MAKEFUN_INPUT_DIR` (else `/input`),
  writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output/`).
- `LICENSE` — MIT.

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: the same history always yields the same forecast.
