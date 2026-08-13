
import numpy as np
import pandas as pd

import config as C


def _rng():
    return np.random.default_rng(C.RANDOM_SEED)


def _autocorrelated_series(n, rng, rho=0.995, scale=1.0, start=0.0):
    """AR(1) walk used for weather / commodity persistence."""
    out = np.empty(n)
    out[0] = start
    innov = rng.normal(0, scale, n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innov[i]
    return out


def build_dataset() -> pd.DataFrame:
    rng = _rng()
    idx = pd.date_range(C.START_DATE, C.END_DATE, freq=C.FREQ, inclusive="left")
    n = len(idx)
    df = pd.DataFrame(index=idx)

    # --- calendar helpers -------------------------------------------------
    hour = idx.hour + idx.minute / 60.0
    doy = idx.dayofyear
    year_frac = (doy - 1) / 365.0
    is_weekday = (idx.dayofweek < 5).astype(float)
    # Rough GB public holidays (lower demand); small illustrative set
    holidays = pd.to_datetime([
        "2023-01-01", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-29",
        "2023-08-28", "2023-12-25", "2023-12-26",
        "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
        "2024-08-26", "2024-12-25", "2024-12-26",
    ])
    is_holiday = idx.normalize().isin(holidays).astype(float)

    # --- weather ----------------------------------------------------------
    # Temperature: annual cycle (cold winter) + diurnal + persistent noise
    annual_temp = 10.5 - 8.0 * np.cos(2 * np.pi * year_frac)      # ~2.5C..18.5C
    diurnal_temp = 3.0 * np.sin(2 * np.pi * (hour - 8) / 24.0)
    temp_noise = _autocorrelated_series(n, rng, rho=0.999, scale=0.12)
    temperature = annual_temp + diurnal_temp + temp_noise
    df["temperature_c"] = temperature

    # Wind capacity factor: persistent, seasonally windier in winter
    wind_lat = _autocorrelated_series(n, rng, rho=0.997, scale=0.028)
    wind_seasonal = 0.08 * np.cos(2 * np.pi * year_frac)          # windier winter
    wind_cf = 0.36 + wind_seasonal + wind_lat
    wind_cf = np.clip(wind_cf, 0.0, 0.92)
    df["wind_cf"] = wind_cf
    df["wind_gen_mw"] = wind_cf * C.WIND_CAPACITY

    # Solar: daytime only, seasonal, weather-modulated
    solar_elev = np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0, None)  # 0 at night
    solar_seasonal = 0.35 + 0.65 * np.clip(np.cos(2 * np.pi * (year_frac - 0.5)), 0, None)
    cloud = np.clip(0.7 + _autocorrelated_series(n, rng, rho=0.995, scale=0.05), 0.1, 1.0)
    solar_cf = solar_elev * solar_seasonal * cloud
    df["solar_gen_mw"] = np.clip(solar_cf, 0, 1) * C.SOLAR_CAPACITY

    # Nuclear: near-flat baseload with occasional outage steps
    nuclear = np.full(n, C.NUCLEAR_BASELOAD)
    n_outages = 6
    for _ in range(n_outages):
        start = rng.integers(0, n - 48 * 20)
        length = rng.integers(48 * 3, 48 * 20)
        nuclear[start:start + length] *= rng.uniform(0.55, 0.8)
    df["nuclear_gen_mw"] = nuclear

    # Interconnector net imports: mean import + noise (acts like -residual demand)
    interconnector = (C.INTERCONNECTOR_MEAN_IMPORT
                      + _autocorrelated_series(n, rng, rho=0.99, scale=40))
    df["interconnector_net_import_mw"] = np.clip(interconnector, -4000, 6000)

    # --- demand -----------------------------------------------------------
    # Double daily peak via two gaussians
    morning = C.DEMAND_MORNING_PEAK * np.exp(-((hour - 8.0) ** 2) / (2 * 1.6 ** 2))
    evening = C.DEMAND_EVENING_PEAK * np.exp(-((hour - 18.0) ** 2) / (2 * 2.0 ** 2))
    night = C.DEMAND_NIGHT_TROUGH * np.exp(-((hour - 4.0) ** 2) / (2 * 3.0 ** 2))
    daily_shape = morning + evening + night

    winter = C.DEMAND_WINTER_UPLIFT * (-np.cos(2 * np.pi * year_frac) * 0.5 + 0.5)
    heating = C.HEATING_SENSITIVITY * np.clip(C.HEATING_THRESHOLD_C - temperature, 0, None)
    weekday_effect = C.DEMAND_WEEKDAY_UPLIFT * is_weekday
    holiday_effect = -3_000.0 * is_holiday
    demand_noise = _autocorrelated_series(n, rng, rho=0.9, scale=90)

    demand = (C.DEMAND_BASE + daily_shape + winter + heating
              + weekday_effect + holiday_effect + demand_noise)
    demand = np.clip(demand, 15_000, 52_000)
    df["national_demand_mw"] = demand

    # --- residual (net) demand -------------------------------------------
    # The single most important short-term price driver: demand net of
    # must-run / zero-marginal-cost supply.
    residual = (demand
                - df["wind_gen_mw"]
                - df["solar_gen_mw"]
                - df["nuclear_gen_mw"]
                - df["interconnector_net_import_mw"])
    df["residual_demand_mw"] = residual

    # --- commodities ------------------------------------------------------
    trend = np.linspace(0, 1, n)
    gas = (C.GAS_START_PTHERM + (C.GAS_END_PTHERM - C.GAS_START_PTHERM) * trend
           + _autocorrelated_series(n, rng, rho=0.9999, scale=0.25))
    gas = np.clip(gas, 30, 300)
    df["gas_price_ptherm"] = gas

    carbon = (C.CARBON_START + (C.CARBON_END - C.CARBON_START) * trend
              + _autocorrelated_series(n, rng, rho=0.9999, scale=0.15))
    df["carbon_price_gbp_t"] = np.clip(carbon, 20, 120)

    # --- price formation --------------------------------------------------
    # 1. Clean spark spread: marginal cost of the CCGT that is usually setting price
    gas_gbp_per_mwh_th = gas * C.PTHERM_TO_GBP_PER_MWH_TH
    ccgt_srmc = (gas_gbp_per_mwh_th / C.CCGT_EFFICIENCY
                 + df["carbon_price_gbp_t"] * C.CCGT_EMISSION_FACTOR / C.CCGT_EFFICIENCY
                 + C.CCGT_VAROM)

    # 2. System tightness drives departure from CCGT SRMC
    tightness = residual / C.DISPATCHABLE_CAPACITY          # ~0.2 .. 1.0+
    # Merit-order / scarcity multiplier: convex in tightness
    scarcity = 0.68 + 0.72 * tightness + 1.9 * np.clip(tightness - 0.85, 0, None) ** 2 * 12
    price = ccgt_srmc * scarcity

    # 3. Renewable oversupply -> negative pricing when residual demand very low
    oversupply = np.clip(1_500 - residual, 0, None)          # MW below a low floor
    price = price - 0.02 * oversupply

    # 4. Idiosyncratic balancing spikes when margin very tight
    spike_prob = np.clip((tightness - 0.9) * 3, 0, 0.6)
    spikes = rng.random(n) < spike_prob
    price = price + spikes * rng.uniform(80, 400, n)

    # 5. Half-hourly noise
    price = price + rng.normal(0, 4.0, n)
    df["price_gbp_mwh"] = price

    # --- expose calendar features used later -----------------------------
    df["is_weekday"] = is_weekday
    df["is_holiday"] = is_holiday
    df["settlement_period"] = ((idx.hour * 60 + idx.minute) // 30 + 1).astype(int)

    df.index.name = "datetime"
    return df


if __name__ == "__main__":
    d = build_dataset()
    print(d[[
        "national_demand_mw", "residual_demand_mw", "wind_gen_mw",
        "solar_gen_mw", "gas_price_ptherm", "price_gbp_mwh",
    ]].describe().round(1).to_string())
    print("\nNegative-price periods: "
          f"{(d['price_gbp_mwh'] < 0).mean() * 100:.2f}%")
    print("Rows:", len(d))
