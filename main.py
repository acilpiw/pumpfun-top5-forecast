#!/usr/bin/env python3
"""Pump.fun top-5 forecaster (pumpfun-top5-forecast-01).

Monte Carlo log-cap diffusion: every coin's log10 market cap (in SOL) is
modelled as a drift + volatility process fitted to its own snapshot series
(shrunk toward the population, drift rate capped because a coin cannot
sustain a multi-x-per-hour climb), extrapolated to each answer read in
task.json, and sampled with a fixed seed.  The place probabilities are the
Monte Carlo frequencies averaged over the reads, scaled down by the share
of top-5 slots a coin born after the cut is expected to take (those slots
are scored from the unnamed mass, so the forecast leaves it there).  The
size distribution of each place is the Monte Carlo histogram of the
log-cap at that place, widened with a broad normal to cover newcomers and
model error.  Standard library only, deterministic, no network.

`build_forecast(task, marks)` is the pure core so a backtest can call it
on a truncated history; `main()` only does the file I/O.
"""
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
N_SAMPLES = 2500
MAX_CANDIDATES = 70
TOP_BY_RANK = 50
TOP_BY_RISE = 20
DRIFT_RATE_CAP = 0.4   # max sustained drift, log10 per hour, either way
DRIFT_CAP = 0.6        # max log10 displacement the drift may add, either way
BASE_SD = 0.5          # base log10 uncertainty even at zero horizon
VOL_LIN = 1.5          # log10 uncertainty growing linearly with horizon
                       # (drift error and jumps; the backtest shows the
                       # spread must grow faster than sqrt(dt))
SD_CAP = 1.2           # cap on the per-read sampling sd
SHRINK_K = 4.0         # shrinkage weight toward the population drift / vol
MIN_P = 0.005          # smallest place probability worth naming a coin at
SIZE_BLEND = 0.25      # weight of the broad normal mixed into the sizes
SIZE_BLEND_SD = 0.5    # sd of that broad normal (log10 SOL)
NEWCOMER_BASE = 0.30   # expected newcomer share of top-5 slots, short horizon
NEWCOMER_PER_HOUR = 0.02
NEWCOMER_MAX = 0.55


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(ts, ys):
    """Least-squares slope and intercept of y on t."""
    n = len(ts)
    mt = sum(ts) / n
    my = sum(ys) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((t - mt) * (y - my) for t, y in zip(ts, ys))
    slope = sxy / sxx
    return slope, my - slope * mt


def bin_of(v, edges):
    """Bin index of log10 cap v: 0 below edges[0], len(edges) at/above last."""
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if v < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def norm_cdf(x, mean, sd):
    if x == -float("inf"):
        return 0.0
    if x == float("inf"):
        return 1.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def build_forecast(task, marks):
    edges = task["size"]["edges"]
    places = int(task["places"])
    reads = [parse_time(r) for r in task["reads"]]
    t0 = parse_time(task["t0"])
    last_time = parse_time(marks[-1]["at"])

    # Per-coin log-cap series (log10 of market cap in SOL).
    series = {}
    for m in marks:
        t = (parse_time(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            try:
                cap = int(r["market_cap_lamports"]) / 1e9
            except (KeyError, TypeError, ValueError):
                continue
            if cap <= 0:
                continue
            s = series.setdefault(r["mint"], {"t": [], "y": []})
            s["t"].append(t)
            s["y"].append(math.log10(cap))
            s["last_rank"] = r["rank"]
            s["symbol"] = r.get("symbol", "")

    # Population drift (log10 per hour) and volatility, from the coins with
    # enough points to fit, then shrink each coin toward them.
    slopes = []
    vols = []
    for s in series.values():
        n = len(s["t"])
        if n >= 2:
            slope, inter = ols(s["t"], s["y"])
            slopes.append(slope)
            if n >= 3:
                resid = [y - (inter + slope * t) for t, y in zip(s["t"], s["y"])]
                vols.append(math.sqrt(sum(e * e for e in resid) / (n - 2)))
    pop_drift = clip(median(slopes), -0.5, 0.5) if slopes else 0.0
    pop_vol = clip(median(vols), 0.05, 1.0) if vols else 0.25

    for s in series.values():
        n = len(s["t"])
        if n >= 2:
            slope, inter = ols(s["t"], s["y"])
            drift = (n * slope + SHRINK_K * pop_drift) / (n + SHRINK_K)
            s["drift"] = clip(drift, -DRIFT_RATE_CAP, DRIFT_RATE_CAP)
            if n >= 3:
                resid = [y - (inter + slope * t) for t, y in zip(s["t"], s["y"])]
                rsd = math.sqrt(sum(e * e for e in resid) / (n - 2))
                s["vol"] = (n * rsd + SHRINK_K * pop_vol) / (n + SHRINK_K)
            else:
                s["vol"] = pop_vol
        else:
            s["drift"] = clip(pop_drift, -DRIFT_RATE_CAP, DRIFT_RATE_CAP)
            s["vol"] = pop_vol
        s["last_y"] = s["y"][-1]

    # Candidates: the largest coins now, plus the fastest risers.
    by_rank = sorted(series.items(), key=lambda kv: kv[1].get("last_rank", 10 ** 9))
    cand_mints = [mint for mint, _ in by_rank[:TOP_BY_RANK]]
    seen = set(cand_mints)
    risers = sorted(
        ((mint, s) for mint, s in series.items() if mint not in seen),
        key=lambda kv: kv[1]["drift"],
        reverse=True,
    )
    for mint, _ in risers[:TOP_BY_RISE]:
        cand_mints.append(mint)
        seen.add(mint)
    cand_mints = cand_mints[:MAX_CANDIDATES]
    cand = [series[m] | {"mint": m} for m in cand_mints]
    n_c = len(cand)
    n_r = len(reads)

    # Per-read horizon and newcomer share.
    dts = []
    for rd in reads:
        dt = (rd - last_time).total_seconds() / 3600.0
        dts.append(max(dt, 0.0))
    mean_dt = sum(dts) / n_r if n_r else 0.0
    newcomer = clip(NEWCOMER_BASE + NEWCOMER_PER_HOUR * mean_dt,
                    NEWCOMER_BASE, NEWCOMER_MAX)
    keep = 1.0 - newcomer

    # Per candidate, per read: extrapolated mean and sd of log-cap.
    means = []
    sds = []
    for s in cand:
        row_m = []
        row_s = []
        for dt in dts:
            disp = clip(s["drift"] * dt, -DRIFT_CAP, DRIFT_CAP)
            row_m.append(s["last_y"] + disp)
            # Uncertainty: base + the coin's own diffusion + a term that
            # grows linearly with horizon (drift error, jumps).  The
            # backtest shows the spread must grow faster than sqrt(dt).
            var = BASE_SD ** 2 + s["vol"] ** 2 * dt + (VOL_LIN * dt) ** 2
            row_s.append(min(math.sqrt(var), SD_CAP))
        means.append(row_m)
        sds.append(row_s)

    # Monte Carlo: one standard-normal draw per coin per iteration, reused
    # across reads (the read's mean/sd scale it), top 5 tracked per read.
    rng = random.Random(SEED)
    place_counts = [[dict() for _ in range(places)] for _ in range(n_r)]
    size_samples = [[[] for _ in range(places)] for _ in range(n_r)]
    for _ in range(N_SAMPLES):
        zs = [rng.gauss(0.0, 1.0) for _ in range(n_c)]
        for r in range(n_r):
            top = []  # (value, coin idx), descending, at most `places`
            for c in range(n_c):
                v = means[c][r] + sds[c][r] * zs[c]
                if len(top) < places or v > top[-1][0]:
                    top.append((v, c))
                    top.sort(reverse=True)
                    del top[places:]
            for place, (v, c) in enumerate(top):
                mint = cand[c]["mint"]
                d = place_counts[r][place]
                d[mint] = d.get(mint, 0) + 1
                size_samples[r][place].append(v)

    # Average the per-read distributions across reads (the one forecast is
    # scored at every read, so the mixture is the honest single forecast).
    probs = [dict() for _ in range(places)]
    for r in range(n_r):
        for place in range(places):
            for mint, cnt in place_counts[r][place].items():
                probs[place][mint] = probs[place].get(mint, 0.0) + cnt / N_SAMPLES / n_r

    # Expected rank and total top-5 probability per coin, for the top5.
    exp_rank = {}
    top5_mass = {}
    for r in range(n_r):
        for place in range(places):
            for mint, cnt in place_counts[r][place].items():
                w = cnt / N_SAMPLES / n_r
                exp_rank[mint] = exp_rank.get(mint, 0.0) + (place + 1) * w
                top5_mass[mint] = top5_mass.get(mint, 0.0) + w

    probabilities = []
    for place in range(places):
        rows = []
        for mint, p in probs[place].items():
            p = p * keep
            if p >= MIN_P:
                rows.append((mint, p))
        rows.sort(key=lambda kv: -kv[1])
        probabilities.extend(
            {"mint": mint, "place": place + 1, "p": round(p, 6)}
            for mint, p in rows[:100]
        )

    # Sizes: Monte Carlo histogram per place, averaged over reads, blended
    # with a broad normal around the place's median log-cap.
    sizes = []
    nbins = len(edges) + 1
    for place in range(places):
        hist = [0.0] * nbins
        med_vals = []
        for r in range(n_r):
            vals = size_samples[r][place]
            if not vals:
                continue
            med_vals.extend(vals)
            for v in vals:
                hist[bin_of(v, edges)] += 1.0 / len(vals) / n_r
        if med_vals:
            med = median(med_vals)
            for b in range(nbins):
                lo = -float("inf") if b == 0 else edges[b - 1]
                hi = float("inf") if b == nbins - 1 else edges[b]
                p_norm = norm_cdf(hi, med, SIZE_BLEND_SD) - norm_cdf(lo, med, SIZE_BLEND_SD)
                hist[b] = (1.0 - SIZE_BLEND) * hist[b] + SIZE_BLEND * p_norm
        for b, p in enumerate(hist):
            if p > 0.0005:
                sizes.append({"place": place + 1, "bin": b, "p": round(p, 6)})

    # top5: the coins most likely to be in the top 5 at all, ordered by
    # expected rank.
    ranked = sorted(top5_mass.items(), key=lambda kv: (-kv[1], exp_rank.get(kv[0], 99)))
    top5 = [mint for mint, _ in ranked[:places]]
    if len(top5) < places:
        for mint, _ in by_rank:
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    out = build_forecast(task, marks)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
