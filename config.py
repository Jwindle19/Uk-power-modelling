"""
Configuration and structural constants for the GB electricity market model.

These constants encode the real structure of the GB (Great Britain) wholesale
electricity market so that both the synthetic data generator and the feature
engineering reflect how prices actually form.

References for the real quantities (approximate, 2023-24 GB system):
  - Settlement periods:      48 half-hourly periods per day (Elexon settlement)
  - National demand range:   ~18 GW (summer night) to ~45 GW (winter peak)
  - Installed wind capacity: ~28 GW (on + offshore)
  - Embedded solar:          ~15 GW
  - Nuclear baseload:        ~4-5 GW
  - Marginal plant:          CCGT (combined-cycle gas) most of the time,
                             so wholesale price tracks gas + carbon via the
                             "clean spark spread".
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Time span for the synthetic dataset
# ---------------------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
FREQ = "30min"                 # GB settlement period granularity
SETTLEMENT_PERIODS_PER_DAY = 48
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Demand structure (MW)
# ---------------------------------------------------------------------------
DEMAND_BASE = 24_000.0         # mean-ish national demand before shape/weather
DEMAND_WINTER_UPLIFT = 3_000.0 # residual winter effect (heating handled below)
DEMAND_WEEKDAY_UPLIFT = 2_500.0
DEMAND_MORNING_PEAK = 4_500.0  # ~08:00
DEMAND_EVENING_PEAK = 6_000.0  # ~18:00
DEMAND_NIGHT_TROUGH = -7_000.0
# Heating sensitivity: MW increase per 1 C below the heating threshold
HEATING_THRESHOLD_C = 15.5
HEATING_SENSITIVITY = 550.0    # MW per degree-C below threshold

# ---------------------------------------------------------------------------
# Renewable / low-marginal-cost supply capacities (MW)
# ---------------------------------------------------------------------------
WIND_CAPACITY = 28_000.0
SOLAR_CAPACITY = 15_000.0
NUCLEAR_BASELOAD = 4_800.0
# Net interconnector imports typically act like negative residual demand
INTERCONNECTOR_MEAN_IMPORT = 3_000.0

# Total dispatchable (mostly CCGT/OCGT/biomass/storage) headroom that must
# balance residual demand. Used to derive system tightness -> scarcity pricing.
DISPATCHABLE_CAPACITY = 34_000.0

# ---------------------------------------------------------------------------
# Commodity price structure
# ---------------------------------------------------------------------------
# Gas in p/therm; converted to GBP/MWh_thermal in the price model.
GAS_START_PTHERM = 130.0       # early-2023 elevated post-crisis level
GAS_END_PTHERM = 75.0          # eased through 2024
CARBON_START = 75.0            # UK ETS GBP/tCO2
CARBON_END = 45.0

# CCGT physics for clean spark spread
CCGT_EFFICIENCY = 0.50         # electrical efficiency (HHV approx)
CCGT_EMISSION_FACTOR = 0.184   # tCO2 per MWh of gas burned
CCGT_VAROM = 3.0               # variable O&M, GBP/MWh
PTHERM_TO_GBP_PER_MWH_TH = 0.341  # 1 p/therm -> GBP/MWh thermal (approx)
