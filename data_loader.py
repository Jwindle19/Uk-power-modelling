"""
Data loading layer.

By default this returns the structurally faithful synthetic dataset so the whole
pipeline runs with no external dependencies. To use REAL GB market data, will be implememnting claude code assisted generation.

Where the the real data (all free) will come from:
  - Elexon BMRS / Insights Solution (https://bmrs.elexon.co.uk):
        * System price / imbalance price  -> price_gbp_mwh
        * Generation by fuel type (FUELINST) -> wind_gen_mw, solar, nuclear, ...
        * National demand (INDO/ITSDO)    -> national_demand_mw
        * Interconnector flows            -> interconnector_net_import_mw
  - NESO Data Portal (https://www.neso.energy/data-portal):
        * Historic demand & embedded solar/wind estimates
  - Day-ahead / system prices: EPEX, Nord Pool, or Elexon MID (market index data)
  - Gas (NBP) and UK ETS carbon prices: commercial feeds or public daily marks

The loader is deliberately tolerant: any REQUIRED_COLUMNS missing from a real
file are reconstructed where possible (e.g. residual demand) or flagged.
"""

from pathlib import Path
import pandas as pd

import config as C
from data_simulator import build_dataset

REQUIRED_COLUMNS = [
    "price_gbp_mwh",
    "national_demand_mw",
    "wind_gen_mw",
    "solar_gen_mw",
    "nuclear_gen_mw",
    "interconnector_net_import_mw",
    "gas_price_ptherm",
    "carbon_price_gbp_t",
    "temperature_c",
]

DEFAULT_REAL_PATH = C.ROOT / "data" / "market_data.csv"


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate columns the analysis expects if a real file omits them."""
    if "residual_demand_mw" not in df:
        df["residual_demand_mw"] = (
            df["national_demand_mw"]
            - df.get("wind_gen_mw", 0)
            - df.get("solar_gen_mw", 0)
            - df.get("nuclear_gen_mw", 0)
            - df.get("interconnector_net_import_mw", 0)
        )
    if "is_weekday" not in df:
        df["is_weekday"] = (df.index.dayofweek < 5).astype(float)
    if "is_holiday" not in df:
        df["is_holiday"] = 0.0
    if "settlement_period" not in df:
        df["settlement_period"] = (
            (df.index.hour * 60 + df.index.minute) // 30 + 1
        ).astype(int)
    if "wind_cf" not in df:
        df["wind_cf"] = df["wind_gen_mw"] / C.WIND_CAPACITY
    return df


def load_data(use_real: bool = False, path: Path | str | None = None) -> pd.DataFrame:
    """Return a clean half-hourly dataframe indexed by datetime."""
    if use_real:
        path = Path(path) if path else DEFAULT_REAL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Real-data mode requested but no file at {path}. "
                "See data_loader.py docstring for how to build it, or run with "
                "use_real=False for the synthetic dataset."
            )
        df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"[loader] WARNING: real file missing columns {missing}; "
                  "downstream features relying on them will be skipped.")
        df = _add_derived(df)
    else:
        df = build_dataset()

    # Basic hygiene shared by both paths
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.interpolate(limit=4).dropna(subset=["price_gbp_mwh"])
    return df


if __name__ == "__main__":
    d = load_data()
    print(f"Loaded {len(d):,} half-hourly rows, "
          f"{d.index.min()} -> {d.index.max()}")
    print("Columns:", list(d.columns))
