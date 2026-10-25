"""Is post-earnings announcement drift still there, and for whom?

Earnings announcements from January 2023 to October 2026 for every US-listed company in the calendar. Two signals,
both known by the close of the day after the announcement: the standardized earnings surprise (SUE, seasonal random walk)
and the earnings announcement return (EAR, the stock's market-adjusted move over days -1..+1). The outcome is the
market-adjusted return over the next 60 trading days (+2..+61), by signal quintile and company size, in event time and as
a calendar-time long-short portfolio after illustrative trading costs.
"""
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import PercentFormatter

NAVY, GOLD, GREY, RED, BLUE, GREEN = "#1F3A5F", "#9A7F2E", "#8A8880", "#B5473A", "#3A6A9A", "#4E7D5B"
SIZE_LABELS = ["micro", "small", "mid", "large"]


# ---------------------------------------------------------------- data
def load_events(cfg):
    c = duckdb.connect()
    ev = c.sql(f"""select c.symbol, c.report_date::date ann, c.fiscal_quarter_ending::date fq
                   from '{cfg['calendar']}' c where c.report_date >= '{cfg['start']}'""").df()
    eps = c.sql(f"select symbol, report_date::date fq, eps::double eps from '{cfg['eps']}' where eps is not null").df()
    sh = c.sql(f"select symbol, report_date::date d, shares_outstanding::double shares from '{cfg['shares']}' where shares_outstanding>0").df()
    ev = ev.drop_duplicates(["symbol", "fq"]).drop_duplicates(["symbol", "ann"])
    return ev, eps, sh


def sue_table(eps, min_hist=4):
    """SUE_t = (EPS_t - EPS_{t-4}) / std of the previous up-to-8 seasonal differences. Uses only past quarters."""
    out = []
    for sym, g in eps.sort_values("fq").groupby("symbol"):
        g = g.drop_duplicates("fq").reset_index(drop=True)
        fq = g.fq.values.astype("datetime64[D]"); e = g.eps.values
        d = np.full(len(g), np.nan)
        for i in range(len(g)):
            # the same fiscal quarter one year earlier (within +/- 20 days)
            j = np.where(np.abs((fq[i] - fq[:i]).astype(int) - 365) <= 20)[0]
            if len(j): d[i] = e[i] - e[j[-1]]
        for i in range(len(g)):
            prev = d[max(0, i - 8):i]; prev = prev[~np.isnan(prev)]
            if np.isnan(d[i]) or len(prev) < min_hist: continue
            s = prev.std(ddof=1)
            if s > 0: out.append((sym, g.fq[i], d[i] / s))
    return pd.DataFrame(out, columns=["symbol", "fq", "sue"])


def load_prices(cfg, symbols):
    c = duckdb.connect(); c.sql("set enable_progress_bar=false")
    syms = ",".join(repr(s) for s in sorted(set(symbols) | {cfg["benchmark"]}))
    px = c.sql(f"""select symbol, report_date::date d, close::double as close, volume::double as vol from '{cfg['prices']}'
                   where report_date >= '2022-09-01' and symbol in ({syms})""").df()
    close = px.pivot(index="d", columns="symbol", values="close").sort_index()
    vol = px.pivot(index="d", columns="symbol", values="vol").reindex(close.index)
    return close, vol


# ---------------------------------------------------------------- events
def build_events(cfg, ev, sue, sh, close, vol):
    dates = close.index.values.astype("datetime64[D]")
    C = close.values; V = vol.values; col = {s: i for i, s in enumerate(close.columns)}
    b = col[cfg["benchmark"]]
    r0, r1 = cfg["reaction_window"]; h0, H = cfg["hold_start"], cfg["hold_days"]
    ev = ev.merge(sue, on=["symbol", "fq"], how="left")
    sh = sh.sort_values("d"); shg = {s: g for s, g in sh.groupby("symbol")}
    rows = []
    for r in ev.itertuples():
        j = col.get(r.symbol)
        if j is None: continue
        d0 = np.searchsorted(dates, np.datetime64(r.ann))          # first trading day on/after the announcement
        if d0 - 21 < 0 or d0 + h0 >= len(dates): continue
        p_m2 = C[d0 - 2, j]
        if not np.isfinite(p_m2) or p_m2 < cfg["min_price"]: continue
        dv = np.nanmedian(C[d0 - 21:d0 - 1, j] * V[d0 - 21:d0 - 1, j])
        if not np.isfinite(dv) or dv < cfg["min_dollar_volume"]: continue
        g = shg.get(r.symbol); mcap = np.nan
        if g is not None:
            gg = g[g.d < pd.Timestamp(r.ann)]
            if len(gg): mcap = gg.shares.iloc[-1] * p_m2
        pa, pb = C[d0 + r0 - 1, j], C[d0 + r1, j]                    # close before day -1 to close of day +1
        ear = pb / pa - (C[d0 + r1, b] / C[d0 + r0 - 1, b]) if np.isfinite(pa) and np.isfinite(pb) else np.nan
        end = min(d0 + h0 + H - 1, len(dates) - 1)
        e0 = C[d0 + h0 - 1, j]; e1 = C[end, j]
        car = e1 / e0 - C[end, b] / C[d0 + h0 - 1, b] if np.isfinite(e0) and np.isfinite(e1) else np.nan
        rows.append(dict(symbol=r.symbol, ann=pd.Timestamp(r.ann), d0=int(d0), sue=r.sue, ear=ear, car=car,
                         full=bool(end == d0 + h0 + H - 1), mcap=mcap, price=p_m2, dollar_vol=dv))
    E = pd.DataFrame(rows)
    E["size"] = pd.cut(E.mcap, [0] + cfg["size_groups"] + [np.inf], labels=SIZE_LABELS)
    return E


def assign_quintiles(E, col, lookback_days):
    """Quintile using breakpoints from announcements in the previous `lookback_days` calendar days only (no look-ahead)."""
    E = E.sort_values("ann").reset_index(drop=True); q = np.full(len(E), np.nan)
    t = E.ann.values; x = E[col].values
    for i in range(len(E)):
        m = (t < t[i]) & (t >= t[i] - np.timedelta64(lookback_days, "D")) & np.isfinite(x)
        if m.sum() < 500 or not np.isfinite(x[i]): continue
        bp = np.quantile(x[m], [0.2, 0.4, 0.6, 0.8]); q[i] = 1 + np.searchsorted(bp, x[i], side="right")
    E[col + "_q"] = q
    return E


# ---------------------------------------------------------------- tests
def month_t(series_by_month):  # clustered by announcement calendar quarter
    s = series_by_month.dropna()
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 else np.nan


def event_time(E, col):
    F = E[E.full & E[col + "_q"].notna() & E.car.notna()].copy(); F["m"] = F.ann.dt.to_period("Q")
    tab = F.groupby(col + "_q").car.agg(["mean", "count"])
    ls = F.groupby("m").apply(lambda g: g[g[col + "_q"] == 5].car.mean() - g[g[col + "_q"] == 1].car.mean())
    by_size = {}
    for s in SIZE_LABELS:
        G = F[F["size"] == s]
        if len(G) < 200: continue
        lss = G.groupby("m").apply(lambda g: g[g[col + "_q"] == 5].car.mean() - g[g[col + "_q"] == 1].car.mean())
        by_size[s] = {"n": int(len(G)), "q5_minus_q1": float(G[G[col + "_q"] == 5].car.mean() - G[G[col + "_q"] == 1].car.mean()),
                      "q5_minus_q1_median": float(G[G[col + "_q"] == 5].car.median() - G[G[col + "_q"] == 1].car.median()),
                      "t_months": month_t(lss)}
    return {"n": int(len(F)), "quintile_car": {int(k): float(v) for k, v in tab["mean"].items()},
            "q5_minus_q1": float(tab["mean"].iloc[-1] - tab["mean"].iloc[0]), "t_months": month_t(ls), "by_size": by_size}


def calendar_portfolio(E, close, cfg, col, size=None, cost=None):
    """Each day hold Q5 long and Q1 short, equal weight, every name in its +2..+61 window. Market-adjusted returns."""
    R = close.pct_change(); b = R[cfg["benchmark"]].values; Rv = R.values; colix = {s: i for i, s in enumerate(close.columns)}
    T = len(R); h0, H = cfg["hold_start"], cfg["hold_days"]
    sel = E[E[col + "_q"].isin([1, 5])]
    if size is not None: sel = sel[sel["size"].isin(size if isinstance(size, (list, tuple)) else [size])]
    out = {}
    for q in (1, 5):
        S = sel[sel[col + "_q"] == q]
        num = np.zeros(T); cnt = np.zeros(T); trades = np.zeros(T)
        for r in S.itertuples():
            j = colix[r.symbol]; a = r.d0 + h0; z = min(r.d0 + h0 + H - 1, T - 1)
            x = Rv[a:z + 1, j] - b[a:z + 1]; ok = np.isfinite(x)
            num[a:z + 1][ok] += x[ok]; cnt[a:z + 1][ok] += 1
            trades[a] += 1
            if z < T - 1: trades[z] += 1
        ret = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        cst = np.where(cnt > 0, trades / np.maximum(cnt, 1), 0.0)
        out[q] = (ret, cst, cnt)
    gross = out[5][0] - out[1][0]
    c = (cost if cost is not None else 0.0) / 1e4
    net = gross - c * (out[5][1] + out[1][1])
    idx = close.index
    ok = (out[5][2] >= 20) & (out[1][2] >= 20)
    return pd.DataFrame({"gross": gross, "net": net}, index=idx)[ok]


def perf(s):
    s = s.dropna()
    if len(s) < 60: return None
    ann = (1 + s).prod() ** (252 / len(s)) - 1; vol = s.std() * np.sqrt(252)
    w = (1 + s).cumprod(); dd = (w / w.cummax() - 1).min()
    return {"ann_return": float(ann), "ann_vol": float(vol), "sharpe": float(s.mean() / s.std() * np.sqrt(252)),
            "max_drawdown": float(dd), "days": int(len(s)), "from": str(s.index[0].date()), "to": str(s.index[-1].date())}


# ---------------------------------------------------------------- figures
def figures(cfg, E, close, out, ports):
    fd = Path(cfg["out_dir"]) / "figures"; fd.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # 1. event-time cumulative abnormal return by SUE quintile
    R = close.pct_change(); b = R[cfg["benchmark"]].values; Rv = R.values; colix = {s: i for i, s in enumerate(close.columns)}
    days = np.arange(-5, 62); paths = {}
    F = E[E.full & E.sue_q.notna()]
    for q in (1, 2, 3, 4, 5):
        acc = []
        for r in F[F.sue_q == q].itertuples():
            j = colix[r.symbol]; a, z = r.d0 - 5, r.d0 + 61
            x = Rv[a:z + 1, j] - b[a:z + 1]
            if len(x) == len(days) and np.isfinite(x).all(): acc.append(np.cumsum(x))
        paths[q] = np.mean(acc, axis=0) if acc else None
    fig, ax = plt.subplots(figsize=(10, 5.4)); cols = [RED, "#C98A6B", GREY, "#6E8DB0", NAVY]
    for q in (1, 2, 3, 4, 5):
        if paths[q] is not None: ax.plot(days, paths[q] * 100, color=cols[q - 1], lw=2.4 if q in (1, 5) else 1.4, label=f"SUE quintile {q}" + (" (worst surprises)" if q == 1 else " (best surprises)" if q == 5 else ""))
    ax.axvline(0, color=GREY, lw=0.8); ax.axvspan(2, 61, color=GOLD, alpha=0.07); ax.axhline(0, color=GREY, lw=0.6)
    ax.text(31, ax.get_ylim()[1] * 0.92, "drift window, days +2 to +61", ha="center", fontsize=9, color=GOLD)
    ax.set_xlabel("Trading days from the announcement"); ax.set_ylabel("Cumulative return vs S&P 500 (%)")
    ax.set_title("Most of the move comes at the announcement, then the gap keeps widening", loc="left", fontsize=12); ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(fd / "event_time_by_sue.png", dpi=160); plt.close(fig)
    # 2. Q5-Q1 drift by size, SUE and EAR
    fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(SIZE_LABELS)); w = 0.36
    for k, (sig, c) in enumerate([("sue", NAVY), ("ear", GOLD)]):
        v = [out["event_time"][sig]["by_size"].get(s, {}).get("q5_minus_q1", np.nan) * 100 for s in SIZE_LABELS]
        bars = ax.bar(x + (k - 0.5) * w, v, w, color=c, label="Earnings surprise (SUE)" if sig == "sue" else "Announcement return (EAR)")
        for bb, val in zip(bars, v):
            if np.isfinite(val): ax.text(bb.get_x() + bb.get_width() / 2, val + (0.12 if val >= 0 else -0.45), f"{val:+.1f}", ha="center", fontsize=9)
    ax.axhline(0, color=GREY, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["Micro\n< $300M", "Small\n$300M–2B", "Mid\n$2–10B", "Large\n> $10B"])
    ax.set_ylabel("Top minus bottom quintile,\n60-day return vs S&P 500 (pp)"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Every size group drifted; micro caps most, but noisily", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(fd / "drift_by_size.png", dpi=160); plt.close(fig)
    # 3. calendar-time growth of $1, long-short by size (net)
    fig, ax = plt.subplots(figsize=(10, 5))
    for lab, c in [("liquid", NAVY), ("small", GOLD)]:
        p = ports.get(("sue", lab))
        if p is not None and len(p): ax.plot(p.index, (1 + p.net).cumprod(), color=c, lw=2, label={"liquid": "Mid and large caps (10 bp)", "micro": "Micro caps (50 bp)", "small": "Small caps (25 bp)"}[lab])
    ax.axhline(1, color=GREY, lw=0.6); ax.set_ylabel("Growth of $1, after costs")
    ax.set_title("Long best surprises, short worst: after trading costs", loc="left", fontsize=12); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(fd / "calendar_long_short.png", dpi=160); plt.close(fig)


def fig_costs(cfg, out):
    L = out["calendar"]["sue_liquid"]; fd = Path(cfg["out_dir"]) / "figures"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    lab = ["No costs", "10 bp", "20 bp", "30 bp"]
    v = [L["gross"]["ann_return"], L["net"]["ann_return"], L["net_at_20bp"]["ann_return"], L["net_at_30bp"]["ann_return"]]
    bars = a1.bar(lab, [x * 100 for x in v], color=[GREY, NAVY, BLUE, "#9DB3CC"])
    for bb, x in zip(bars, v): a1.text(bb.get_x() + bb.get_width() / 2, x * 100 + 0.15, f"{x*100:.1f}%", ha="center", fontsize=10)
    a1.set_ylabel("Annual return, % (long-short)"); a1.set_xlabel("Cost per trade, one way"); a1.set_title("Costs decide the answer", loc="left", fontsize=12)
    yrs = list(L["by_year_net"].keys()); yv = [L["by_year_net"][y] * 100 for y in yrs]
    bars = a2.bar([y if y != "2026" else "2026 YTD" for y in yrs], yv, color=[RED if x < 0 else NAVY for x in yv])
    for bb, x in zip(bars, yv): a2.text(bb.get_x() + bb.get_width() / 2, x + (0.3 if x >= 0 else -1.0), f"{x:+.1f}%", ha="center", fontsize=10)
    a2.axhline(0, color=GREY, lw=0.8); a2.set_ylabel("Return after 10 bp costs, %"); a2.set_title("And the returns are lumpy", loc="left", fontsize=12)
    fig.suptitle("Mid- and large-cap earnings-surprise book", x=0.01, ha="left", fontsize=10, color=GREY)
    fig.tight_layout(); fig.savefig(fd / "costs_and_years.png", dpi=160); plt.close(fig)


def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path)); od = Path(cfg["out_dir"]); od.mkdir(exist_ok=True)
    ev, eps, sh = load_events(cfg)
    sue = sue_table(eps, cfg["sue_min_history"])
    close, vol = load_prices(cfg, ev.symbol.unique())
    E = build_events(cfg, ev, sue, sh, close, vol)
    for c in ("sue", "ear"): E = assign_quintiles(E, c, cfg["breakpoint_lookback_days"])
    out = {"n_announcements_raw": int(len(ev)), "n_events": int(len(E)), "n_with_sue": int(E.sue.notna().sum()),
           "n_firms": int(E.symbol.nunique()), "first": str(E.ann.min().date()), "last": str(E.ann.max().date()),
           "size_counts": {k: int(v) for k, v in E["size"].value_counts().items()}}
    out["event_time"] = {c: event_time(E, c) for c in ("sue", "ear")}
    # halves: announcements 2023-24 vs 2025-26
    for nm, part in [("2023_24", E[E.ann < "2025-01-01"]), ("2025_26", E[E.ann >= "2025-01-01"])]:
        out["event_time"]["sue_" + nm] = event_time(part, "sue")
    # combined: top on both vs bottom on both
    F = E[E.full & E.sue_q.notna() & E.ear_q.notna()]
    hi = F[(F.sue_q == 5) & (F.ear_q == 5)].car.mean(); lo = F[(F.sue_q == 1) & (F.ear_q == 1)].car.mean()
    out["event_time"]["both"] = {"n_long": int(((F.sue_q == 5) & (F.ear_q == 5)).sum()), "n_short": int(((F.sue_q == 1) & (F.ear_q == 1)).sum()), "q5_minus_q1": float(hi - lo)}
    # calendar-time portfolios, gross and net of illustrative costs by size group
    cost = dict(zip(["large", "mid", "small", "micro"], [cfg["cost_bps_one_way"][0], cfg["cost_bps_one_way"][0], cfg["cost_bps_one_way"][1], cfg["cost_bps_one_way"][2]]))
    ports = {}; out["calendar"] = {}
    for sig in ("sue", "ear"):
        for s in ["all"] + SIZE_LABELS:
            c = 25 if s == "all" else cost[s]
            p = calendar_portfolio(E, close, cfg, sig, None if s == "all" else s, c); ports[(sig, s)] = p
            out["calendar"][f"{sig}_{s}"] = {"cost_bps_one_way": c, "gross": perf(p.gross), "net": perf(p.net)}
    # the liquid book: mid and large caps together, 10 bp one way
    p = calendar_portfolio(E, close, cfg, "sue", ["mid", "large"], cfg["cost_bps_one_way"][0]); ports[("sue", "liquid")] = p
    out["calendar"]["sue_liquid"] = {"cost_bps_one_way": cfg["cost_bps_one_way"][0], "gross": perf(p.gross), "net": perf(p.net),
        "by_year_net": {str(y): float((1 + g.net).prod() - 1) for y, g in p.groupby(p.index.year)}}
    for c2 in (20, 30):
        out["calendar"]["sue_liquid"][f"net_at_{c2}bp"] = perf(p.gross - (p.gross - p.net) * c2 / cfg["cost_bps_one_way"][0])
    ln = out["calendar"]["sue_liquid"]["net"]
    out["illustration"] = {"book_usd": cfg["book_usd"], "liquid_net_pnl_per_year_usd": ln["ann_return"] * cfg["book_usd"]}
    E.to_csv(od / "events.csv", index=False)
    json.dump(out, open(od / "metrics.json", "w"), indent=1, default=float)
    figures(cfg, E, close, out, ports)
    fig_costs(cfg, out)
    print(json.dumps({k: v for k, v in out.items() if k != "calendar"}, indent=1, default=float)[:4000])
    for k, v in out["calendar"].items():
        print(k, {a: (None if b is None else {x: round(y, 4) if isinstance(y, float) else y for x, y in b.items() if x in ("ann_return", "sharpe", "max_drawdown")}) for a, b in v.items() if a != "cost_bps_one_way"})


if __name__ == "__main__":
    main()
