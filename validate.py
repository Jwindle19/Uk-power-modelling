"""
Sanity checks that run before the analysis, so a bad data pull fails overtly
instead of quietly producing a wrong report. Two kinds of check:

- data checks: which are the columns there, in plausible ranges, not full of gaps?
- model checks: does the fitted OLS actually behave like the merit order should
  (right-signed coefficients, multicollinearity not blowing up)? If it doesn't,
  something upstream is wrong with the data.

Nothing here is fatal by default - run_analysis prints the report and carries on
for the synthetic set - but on real data a FAIL is your cue to go look.

Why OLS for validation?
-----------------------
I'm still gaining familiarity with standard econometric models, but OLS is super 
handy as a quick pre-flight sanity check. 

Unlike complex black-box ML models, OLS gives you direct positive or negative 
coefficients right out of the box. That means you can immediately check if the basic 
grid physics and market rules actually make sense—like making sure wind/solar drop 
the price while demand pushes it up. Plus, the closed-form math runs almost instantly, 
and checking VIFs (Variance Inflation Factors) catches bad or collinear data before 
we feed it into the heavier gradient boosting models downstream.

References & Methodology Sources
--------------------------------
[1] Kirschen, D. S., & Strbac, G. (2018). *Fundamentals of Power System Economics* (2nd ed.).
    Wiley. 
    --> Source for market clearing, merit-order effect dynamics, and marginal plant economics
[2] Wooldridge, J. M. (2012). *Introductory Econometrics: A Modern Approach* (5th ed.).
    Cengage Learning.
    --> Source for linear regression fundamentals, coefficient interpretation, and the 
        VIF < 10 threshold rule for diagnosing multicollinearity.
[3] McKinney, W. (2022). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter* (3rd ed.).
    O'Reilly Media.
    --> Source for Python data processing patterns, vectorized range-checking routines, and 
        handling missingness/outliers using Pandas and NumPy.
"""

import numpy as np

# plausible ranges for GB half-hourly data. Wide on purpose: these are meant to
# catch unit mix-ups and broken pulls, not to second-guess a real market.
RANGES = {
    "price_gbp_mwh": (-500, 4000),
    "national_demand_mw": (10_000, 60_000),
    "wind_gen_mw": (0, 30_000),
    "solar_gen_mw": (0, 16_000),
    "gas_price_ptherm": (10, 600),
    "carbon_price_gbp_t": (0, 200),
    "temperature_c": (-20, 45),
}

REQUIRED = ["price_gbp_mwh", "national_demand_mw", "wind_gen_mw",
            "residual_demand_mw"]


def _check(name, ok, detail=""):
    return {"check": name, "pass": bool(ok), "detail": detail}


def validate_data(df):
    """Column presence, ranges, and missingness. Returns a list of check dicts."""
    results = []

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    results.append(_check(
        "required columns present",
        not missing_cols,
        "missing: " + ", ".join(missing_cols) if missing_cols else "all present",
    ))

    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        out_of_range = ((s < lo) | (s > hi)).mean() * 100
        results.append(_check(
            f"{col} in [{lo}, {hi}]",
            out_of_range < 1.0,           # allow a tiny fraction of outliers
            f"{out_of_range:.2f}% out of range",
        ))

    gaps = df.isna().mean() * 100
    worst = gaps[gaps > 5].sort_values(ascending=False)
    results.append(_check(
        "missingness under 5% per column",
        worst.empty,
        "OK" if worst.empty else "high gaps: "
        + ", ".join(f"{c} {v:.0f}%" for c, v in worst.items()),
    ))

    return results


def validate_merit_order(df):
    """Fit the interpretable OLS and check it points the right way. On sane data:
    wind and solar coefficients should be negative (merit-order effect), demand
    positive, and no VIF should be sky-high."""
    from models import fit_price_level_ols
    r = fit_price_level_ols(df)
    coef = r["coefficients"]

    checks = [
        _check("wind coefficient negative", coef.get("wind_gen_mw", 0) < 0,
               f"{coef.get('wind_gen_mw', float('nan')):.4f}"),
        _check("solar coefficient negative", coef.get("solar_gen_mw", 0) < 0,
               f"{coef.get('solar_gen_mw', float('nan')):.4f}"),
        _check("demand coefficient positive", coef.get("national_demand_mw", 0) > 0,
               f"{coef.get('national_demand_mw', float('nan')):.4f}"),
        _check("max VIF under 10", r["vif"].max() < 10,
               f"max VIF {r['vif'].max():.1f}"),
        _check("OLS R2 above 0.4", r["r2"] > 0.4, f"R2 {r['r2']:.3f}"),
    ]
    return checks


def run_all(df, verbose=True):
    results = validate_data(df) + validate_merit_order(df)
    n_fail = sum(not r["pass"] for r in results)
    if verbose:
        for r in results:
            flag = "ok  " if r["pass"] else "FAIL"
            print(f"  [{flag}] {r['check']}  ({r['detail']})")
        print(f"  {len(results) - n_fail}/{len(results)} checks passed")
    return results, n_fail


if __name__ == "__main__":
    from data_loader import load_data
    from features import build_features
    d = build_features(load_data())
    run_all(d)
