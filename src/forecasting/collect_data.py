#!/usr/bin/env python
# coding: utf-8
"""
collect_data.py

Full data collection pipeline:
  1. Downloads energy data from NED API (CO2, solar, wind, biomass, etc.)
  2. Downloads weather data from Open-Meteo (no API key needed)
  3. Merges everything into one clean DataFrame
  4. Saves: ned_hourly_filled.csv + openmeteo_historical.csv + master_dataset.csv

Usage:
    python collect_data.py

Output files:
    ned_hourly_filled.csv       — NED energy data, cleaned & filled
    openmeteo_historical.csv    — Weather features (historical)
    master_dataset.csv          — Final merged dataset ready for modeling
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError
import time
from dotenv import load_dotenv
load_dotenv()

# =========================
# CONFIG
# =========================

@dataclass(frozen=True)
class PipelineConfig:
    # API
    ned_url: str = "https://api.ned.nl/v1/utilizations"
    ned_api_key_env: str = "NED_API_KEY"

    @property
    def ned_api_key(self) -> str:
        key = os.getenv(self.ned_api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {self.ned_api_key_env} not set"
            )
        return key
    
    # Dates
    start_date: str = "2023-01-01T00:00:00Z"
    end_date_strictly_before: str = "2026-01-01T00:00:00Z"
    train_end: str = "2025-01-01T00:00:00Z"  # used to avoid leakage in filling

    # Weather
    lat: float = 52.3
    lon: float = 5.3
    timezone: str = "UTC"  # use UTC for all processing to avoid DST issues; 
    weather_vars: Tuple[str, ...] = (
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "cloud_cover",
        "cloud_cover_low",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_100m",
        "wind_gusts_10m",
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
    )

    # Fill limits
    co2_interp_hours: int = 6
    wind_interp_hours: int = 3

    # Alerts
    max_missing_rows_frac_per_day: float = 0.01
    max_imputed_frac_per_day_target: float = 0.01
    max_consecutive_missing_hours_target: int = 6

    # Output
    data_dir_name: str = "data"  # created in repo root (see resolve_data_dir)


# NED type IDs
TYPE_MAP: Dict[str, int] = {
    "co2-factor":       27,
    "solar-energy":      2,
    "wind-energy":       1,
    "biomass-power":    25,
    "waste-power":      21,
    "other-power":      26,
    "fossil-gas-power": 18,
    "fossil-coal":      19,
    "nuclear":          20,
    "wind-offshore":    51,
}


def resolve_repo_root() -> Path:
    """
    Resolve repo root based on this file location:
    repo_root/
      src/forecasting/collect_data.py
    """
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(cfg: PipelineConfig) -> Path:
    data_dir = resolve_repo_root() / cfg.data_dir_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def ned_headers(cfg: PipelineConfig) -> Dict[str, str]:
    api_key = os.environ.get(cfg.ned_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing NED API key. Set environment variable {cfg.ned_api_key_env}."
        )
    return {"X-AUTH-TOKEN": api_key, "accept": "application/ld+json"}


# =========================
# NED API
# =========================

def fetch_all_ned(url: str, headers: Dict[str, str], params: Dict, sleep_s: float = 0.2, max_retries: int = 8, timeout_s: int = 180):
    """Paginated fetch from NED API with retry on 429/5xx."""
    all_records = []
    next_url = url
    next_params = params

    while True:
        r = None
        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.get(next_url, headers=headers, params=next_params, timeout=timeout_s)
            except (ReadTimeout, ConnectTimeout, ConnectionError) as e:
                last_err = e
                time.sleep(min(60, 2 ** attempt))
                continue
            
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", min(60, 2 ** attempt)))
                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:
                time.sleep(min(60, 2 ** attempt))
                continue

            break

        if r is None:
            raise RuntimeError(f"NED request failed after retries. Last error: {last_err!r}")

        if r.status_code != 200:
            print("FULL URL:", r.url)
            print("STATUS:", r.status_code)
            print("BODY:", r.text[:500])
            raise RuntimeError(f"NED HTTP {r.status_code}: {r.text[:500]}")
            
        data = r.json()
        all_records.extend(data.get("hydra:member", []))

        hydra_next = (data.get("hydra:view") or {}).get("hydra:next")
        if not hydra_next:
            break

        next_url = "https://api.ned.nl" + hydra_next
        next_params = None
        time.sleep(sleep_s)

    return all_records


def download_ned(cfg: PipelineConfig, name: str, start_date: str, end_date: str, classification: int = 2, point: int = 0, activity: int = 1) -> pd.DataFrame:
    """Download one NED dataset by name."""
    params = {
        "point":                      point,
        "type":                       TYPE_MAP[name],
        "granularity":                5,   # Hour
        "granularitytimezone":        1,   # CET
        "activity":                   activity,
        "classification":             classification,
        "validfrom[after]":           start_date,
        "validfrom[strictly_before]": end_date,
    }
    records = fetch_all_ned(cfg.ned_url, ned_headers(cfg), params)
    df = pd.DataFrame(records)

    if df.empty:
        print(f"  [NED] WARNING: {name} returned no records!")
        return df

    df["validfrom"] = pd.to_datetime(df["validfrom"], utc=True, errors="coerce")
    df["validto"]   = pd.to_datetime(df["validto"],   utc=True, errors="coerce")

    cols = [c for c in ["validfrom", "validto", "capacity", "volume", "percentage", "emission", "emissionfactor"] if c in df.columns]
    return df[cols].sort_values("validfrom").reset_index(drop=True)


def to_hourly(df: pd.DataFrame, value_cols: List[str], prefix: str) -> pd.DataFrame:
    """Resample to 1H UTC index and add prefix to columns."""
    if df.empty:
        return pd.DataFrame()
    ts = df.set_index("validfrom").sort_index().asfreq("h")
    keep = [c for c in value_cols if c in ts.columns]
    return ts[keep].add_prefix(prefix)


def collect_ned(cfg: PipelineConfig, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Download all NED sources and return hourly DataFrames dict."""
    sources = {
        "co2":     ("co2-factor",       ["emissionfactor", "emission", "volume"], "co2_"),
        "solar":   ("solar-energy",      ["capacity", "volume"],                  "solar_"),
        "wind":    ("wind-energy",       ["capacity", "volume"],                  "wind_"),
        "biomass": ("biomass-power",     ["capacity", "volume"],                  "biomass_"),
        "waste":   ("waste-power",       ["capacity", "volume"],                  "waste_"),
        "gas":     ("fossil-gas-power",  ["capacity", "volume"],                  "gas_"),
        "coal":    ("fossil-coal",       ["capacity", "volume"],                  "coal_"),
        "nuclear": ("nuclear",           ["capacity", "volume"],                  "nucl_"),
        "offwind": ("wind-offshore",     ["capacity", "volume"],                  "offwind_"),
    }

    hourly: Dict[str, pd.DataFrame] = {}
    for key, (ned_name, cols, prefix) in sources.items():
        print(f"  Downloading NED: {ned_name}...")
        df_raw = download_ned(cfg, ned_name, start_date, end_date)
        hourly[key] = to_hourly(df_raw, cols, prefix)

    return hourly

# =========================
# Weather (Open-Meteo)
# =========================

def fetch_openmeteo_historical(cfg: PipelineConfig, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive. No API key required."""
    print("  Downloading Open-Meteo historical weather...")
    params = {
        "latitude":        cfg.lat,
        "longitude":       cfg.lon,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          ",".join(cfg.weather_vars),
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index.name = "validfrom"
    df = df.add_prefix("weather_")
    print(f"  Open-Meteo historical: {len(df)} rows, {df.index.min()} → {df.index.max()}")
    return df


def fetch_openmeteo_forecast(cfg: PipelineConfig, forecast_days: int = 7) -> pd.DataFrame:
    """Fetch live 7-day weather forecast from Open-Meteo (168 steps, 1h resolution)."""
    print("  Downloading Open-Meteo 7-day forecast...")
    params = {
        "latitude":        cfg.lat,
        "longitude":       cfg.lon,
        "hourly":          ",".join(cfg.weather_vars),
        "forecast_days":   forecast_days,
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    }

    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo HTTP {r.status_code} URL={r.url}\nBody: {r.text[:500]}")
    payload = r.json()

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index.name = "validfrom"
    df = df.add_prefix("weather_")
    print(f"  Open-Meteo forecast:   {len(df)} rows, {df.index.min()} → {df.index.max()}")
    return df


# =========================
# Time features
# =========================

def add_time_features(cfg: PipelineConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features and a daylight proxy. Safe for forecasting (known for all future timestamps)."""
    df = df.copy()
    local_idx = df.index.tz_convert(cfg.timezone)

    hour = local_idx.hour
    dow = local_idx.dayofweek
    doy = local_idx.dayofyear
    month = local_idx.month

    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)

    df["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7)

    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    df["sin_month"] = np.sin(2 * np.pi * month / 12)
    df["cos_month"] = np.cos(2 * np.pi * month / 12)

    df["hour"] = hour
    df["dow"] = dow
    df["month"] = month

    # Daylight proxy
    summer_offset = np.cos(2 * np.pi * (doy - 172) / 365.25)
    sunrise = 4.5 + 2.0 * (1 - summer_offset) / 2
    sunset = 21.5 - 2.0 * (1 - summer_offset) / 2
    df["is_daylight"] = ((hour >= sunrise) & (hour <= sunset)).astype("int8")

    return df


# =========================
# Cleaning & imputation
# =========================

def make_master_index(cfg: PipelineConfig) -> pd.DatetimeIndex:
    start_cet = pd.Timestamp(cfg.start_date, tz=cfg.timezone)
    end_cet = pd.Timestamp(cfg.end_date_strictly_before, tz=cfg.timezone)
    idx_cet = pd.date_range(start=start_cet, end=end_cet - pd.Timedelta(hours=1), freq="1H")
    return idx_cet.tz_convert("UTC")


def align_to_master(ts: pd.DataFrame, master_index: pd.DatetimeIndex, name: str) -> pd.DataFrame:
    aligned = ts.reindex(master_index)
    missing = aligned.isna().any(axis=1).sum()
    print(f"  {name}: rows={len(aligned)}, missing_rows_any_col={missing}")
    return aligned


def controlled_fill(cfg: PipelineConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with domain-appropriate strategies."""
    df = df.copy()

    for c in df.columns:
        df[c + "_was_missing"] = df[c].isna().astype("int8")

    if "co2_emissionfactor" in df.columns:
        df["co2_emissionfactor"] = df["co2_emissionfactor"].interpolate(
            method="time",
            limit=cfg.co2_interp_hours,
            limit_direction="forward",
        )

    for c in ["solar_capacity", "solar_volume"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in ["wind_capacity", "wind_volume", "offwind_capacity", "offwind_volume"]:
        if c in df.columns:
            df[c] = df[c].interpolate(
                method="time",
                limit=cfg.wind_interp_hours,
                limit_direction="forward",
            )

    return df


def longest_nan_streak(s: pd.Series) -> int:
    is_nan = s.isna().to_numpy()
    max_streak, cur = 0, 0
    for v in is_nan:
        cur = cur + 1 if v else 0
        max_streak = max(max_streak, cur)
    return max_streak


def run_alerts(cfg: PipelineConfig, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    miss_any = df_before.isna().any(axis=1)
    daily_miss = miss_any.resample("D").mean()
    worst = float(daily_miss.max())

    if worst > cfg.max_missing_rows_frac_per_day:
        bad = daily_miss[daily_miss > cfg.max_missing_rows_frac_per_day]
        print(f"\n[ALERT] Too many missing rows before fill. Worst day: {worst:.2%}")
        print(bad.head(5))

    if "co2_emissionfactor" in df_before.columns:
        streak = longest_nan_streak(df_before["co2_emissionfactor"])
        if streak > cfg.max_consecutive_missing_hours_target:
            print(f"\n[ALERT] CO2 target has {streak}-hour gap before fill!")


# =========================
# Orchestration
# =========================

def build_master_dataset(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline entrypoint for both scripts and notebooks.
    Returns:
      df_master: final merged dataset (timestamp column included)
      df_weather_fc: 7-day forecast weather (separate artifact for inference)
    """
    data_dir = resolve_data_dir(cfg)

    print("=" * 60)
    print("STEP 1: Download NED energy data")
    print("=" * 60)
    hourly = collect_ned(cfg, cfg.start_date, cfg.end_date_strictly_before)

    print("\n" + "=" * 60)
    print("STEP 2: Download Open-Meteo weather data")
    print("=" * 60)
    end_date_inclusive = "2025-12-31T23:00:00Z"  # Open-Meteo expects inclusive end date, so use 1 year minus 1 hour
    df_weather_hist = fetch_openmeteo_historical(cfg, cfg.start_date, end_date_inclusive)
    (data_dir / "openmeteo_historical.csv").write_text("")  # ensure file can be created (optional)
    df_weather_hist.to_csv(data_dir / "openmeteo_historical.csv")
    print(f"  Saved: {data_dir / 'openmeteo_historical.csv'}")

    df_weather_fc = fetch_openmeteo_forecast(cfg, forecast_days=7)
    df_weather_fc.to_csv(data_dir / "openmeteo_forecast_7days.csv")
    print(f"  Saved: {data_dir / 'openmeteo_forecast_7days.csv'}")

    print("\n" + "=" * 60)
    print("STEP 3: Align NED data to master hourly index")
    print("=" * 60)
    master_idx = make_master_index(cfg)

    aligned = {key: align_to_master(ts, master_idx, key) for key, ts in hourly.items() if not ts.empty}
    df_ned = pd.concat(list(aligned.values()), axis=1)
    print(f"\n  NED combined: {len(df_ned)} rows, {df_ned.shape[1]} columns")

    print("\n" + "=" * 60)
    print("STEP 4: Clean & fill NED data (split-before-fill to avoid leakage)")
    print("=" * 60)
    raw_train = df_ned[df_ned.index < cfg.train_end].copy()
    raw_test = df_ned[df_ned.index >= cfg.train_end].copy()

    train_filled = controlled_fill(cfg, raw_train)
    test_filled = controlled_fill(cfg, raw_test)

    df_ned_filled = pd.concat([train_filled, test_filled]).sort_index()
    run_alerts(cfg, df_ned, df_ned_filled)

    df_ned_filled.to_csv(data_dir / "ned_hourly_filled.csv")
    print(f"  Saved: {data_dir / 'ned_hourly_filled.csv'}")

    print("\n" + "=" * 60)
    print("STEP 5: Merge NED + Weather + Time features")
    print("=" * 60)
    df_master = df_ned_filled.join(df_weather_hist, how="left")
    df_master = add_time_features(cfg, df_master)

    n_missing_weather = df_master[[c for c in df_master.columns if c.startswith("weather_")]].isna().any(axis=1).sum()
    print(f"  Master dataset: {len(df_master)} rows, {df_master.shape[1]} columns")
    print(f"  Missing weather rows: {n_missing_weather}")
    print(f"  Target (co2_emissionfactor) NaN: {df_master.get('co2_emissionfactor').isna().sum() if 'co2_emissionfactor' in df_master.columns else 'N/A'}")

    df_master_out = df_master.reset_index().rename(columns={"validfrom": "timestamp"})
    df_master_out.to_csv(data_dir / "master_dataset.csv", index=False)
    print(f"\n  Saved: {data_dir / 'master_dataset.csv'}  ← use this for modeling!")

    return df_master_out, df_weather_fc


def main() -> int:
    cfg = PipelineConfig()
    build_master_dataset(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
