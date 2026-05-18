"""
Synthetic Bike Sharing dataset generator.

Produces a CSV that mirrors the structure of the UCI Bike Sharing Dataset
(archive.ics.uci.edu/dataset/275).  All numeric ranges and correlations are
based on the published dataset statistics so the trained model behaves
realistically in demos.
"""

import numpy as np
import pandas as pd


def generate_bike_sharing(n_rows: int = 17_379, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with the same schema as hour.csv from the UCI dataset."""
    rng = np.random.default_rng(seed)

    # -- calendar skeleton --------------------------------------------------
    # 2 years × ~365 days × 24 hours ≈ 17 376 records (matches UCI)
    hours = np.tile(np.arange(24), n_rows // 24 + 1)[:n_rows]
    days  = np.repeat(np.arange(n_rows // 24 + 1), 24)[:n_rows]

    yr      = (days // 365).clip(0, 1).astype(int)
    mnth    = ((days % 365) // 30 + 1).clip(1, 12).astype(int)
    season  = ((mnth - 1) // 3 + 1).clip(1, 4).astype(int)
    weekday = (days % 7).astype(int)
    holiday = (rng.random(n_rows) < 0.028).astype(int)
    workingday = ((weekday < 5) & (holiday == 0)).astype(int)

    # -- weather ------------------------------------------------------------
    weathersit = rng.choice([1, 2, 3, 4], n_rows, p=[0.47, 0.38, 0.14, 0.01])

    # temp follows a seasonal sinusoid + noise, normalised to [0, 1]
    temp_raw = (
        0.45
        + 0.25 * np.sin(2 * np.pi * (days % 365) / 365 - np.pi / 2)
        + rng.normal(0, 0.07, n_rows)
    ).clip(0, 1)
    atemp = (temp_raw * 0.95 + rng.normal(0, 0.03, n_rows)).clip(0, 1)
    hum   = (
        0.62
        - 0.05 * np.sin(2 * np.pi * (days % 365) / 365)
        + rng.normal(0, 0.12, n_rows)
    ).clip(0, 1)
    windspeed = rng.beta(2, 5, n_rows)

    # -- demand model -------------------------------------------------------
    # Base: hour-of-day demand profile (bimodal commuter pattern)
    hour_profile = np.array([
        0.1, 0.05, 0.03, 0.03, 0.05, 0.15,
        0.35, 0.80, 1.00, 0.60, 0.45, 0.55,
        0.65, 0.55, 0.45, 0.55, 0.75, 0.95,
        0.85, 0.65, 0.50, 0.40, 0.30, 0.20,
    ])
    base = hour_profile[hours] * 300

    # modifiers
    base *= (1 + 0.3 * yr)
    base *= (1 + 0.08 * temp_raw)
    base *= (1 - 0.15 * (weathersit - 1))
    base *= np.where(workingday, 1.0, 0.7)
    base *= np.where(holiday, 0.6, 1.0)
    base *= (1 - 0.2 * windspeed)

    noise = rng.normal(0, 0.12, n_rows)
    cnt   = (base * (1 + noise)).clip(1).round().astype(int)

    # -- assemble -----------------------------------------------------------
    df = pd.DataFrame({
        "season":     season,
        "yr":         yr,
        "mnth":       mnth,
        "hr":         hours,
        "holiday":    holiday,
        "weekday":    weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp":       temp_raw.round(4),
        "atemp":      atemp.round(4),
        "hum":        hum.round(4),
        "windspeed":  windspeed.round(4),
        "cnt":        cnt,
    })
    return df


if __name__ == "__main__":
    df = generate_bike_sharing()
    out = "data/bike_sharing.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} rows → {out}")
    print(df.head())
