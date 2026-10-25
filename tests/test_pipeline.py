import numpy as np
import pandas as pd

from eardrift.pipeline import assign_quintiles, sue_table


def _eps(vals, start="2018-03-31"):
    fq = pd.date_range(start, periods=len(vals), freq="QE")
    return pd.DataFrame({"symbol": "AAA", "fq": fq, "eps": vals})


def test_sue_uses_only_past_quarters():
    e = _eps([1.0, 1.1, 0.9, 1.2, 1.1, 1.3, 1.0, 1.4, 1.2, 1.5, 1.1, 1.6, 1.3, 1.9])
    base = sue_table(e)
    e2 = e.copy(); e2.loc[e2.index[-1], "eps"] = 50.0     # change only the latest quarter
    after = sue_table(e2)
    pd.testing.assert_frame_equal(base.iloc[:-1].reset_index(drop=True), after.iloc[:-1].reset_index(drop=True))
    assert after.sue.iloc[-1] > base.sue.iloc[-1]


def test_sue_needs_history():
    assert sue_table(_eps([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])).empty   # only 2 seasonal changes: below the 4 required


def test_seasonal_difference_scaled_by_past_volatility():
    vals = [1, 1, 1, 1, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 2.1]
    s = sue_table(_eps(vals)); d = np.array(vals); diffs = d[4:] - d[:-4]
    expected = diffs[-1] / diffs[-9:-1].std(ddof=1)
    assert np.isclose(s.sue.iloc[-1], expected)


def test_quintile_breakpoints_ignore_later_announcements():
    rng = np.random.default_rng(0)
    E = pd.DataFrame({"ann": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(2000) // 20, "D"), "sue": rng.normal(size=2000)})
    a = assign_quintiles(E.copy(), "sue", 91)
    E2 = E.copy(); E2.loc[E2.ann > pd.Timestamp("2024-03-15"), "sue"] = 100.0   # future values explode
    b = assign_quintiles(E2, "sue", 91)
    m = a.ann <= pd.Timestamp("2024-03-15")
    assert (a.loc[m, "sue_q"].fillna(0).values == b.loc[m, "sue_q"].fillna(0).values).all()
    assert set(a.sue_q.dropna().unique()) <= {1, 2, 3, 4, 5}
