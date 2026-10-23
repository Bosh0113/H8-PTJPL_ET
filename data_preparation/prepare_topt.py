"""Prepare moving-average meteorological and vegetation inputs for Topt."""

import argparse
import json
import tempfile
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from calculate_topt import FIELDS, decode, save_scaled


def load(path):
    return decode(np.load(path))


def saturation_vapor_pressure(t_c):
    t_c = np.asarray(t_c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pressure = 0.61121 * np.exp(17.502 * t_c / (t_c + 240.97))
    return np.where(
        np.isfinite(t_c) & (t_c > -240.97) & np.isfinite(pressure) & (pressure > 0),
        pressure,
        np.nan,
    )


def hourly_paths(root, day, tag, hour):
    stamp = (day + timedelta(hours=hour)).strftime("%Y%m%d%H%M")
    year = root / str(day.year)
    return [
        year / directory / f"{stamp}_input_{tag}_{suffix}.npy"
        for directory, suffix in [
            ("r_net_modis001", "rnet"),
            ("ta_era001", "ta"),
            ("td_era001", "td"),
        ]
    ]


def daily_means(root, day, tag, shape, skip_missing):
    sums = {field: np.zeros(shape) for field in FIELDS}
    counts = {field: np.zeros(shape, dtype=np.uint8) for field in FIELDS}
    missing = []
    ndvi_path = (
        root
        / str(day.year)
        / "ndvi_daily_ahi001"
        / f"{day:%Y%m%d}0000_daily_{tag}_ndvi.npy"
    )
    if not ndvi_path.exists():
        if not skip_missing:
            raise FileNotFoundError(ndvi_path)
        return {field: np.full(shape, np.nan) for field in FIELDS}, [str(ndvi_path)]
    savi = 0.45 * load(ndvi_path) + 0.132
    if savi.shape != shape:
        raise ValueError(f"Input shape mismatch: {ndvi_path}")
    for hour in range(24):
        paths = hourly_paths(root, day, tag, hour)
        absent = [str(path) for path in paths if not path.exists()]
        if absent:
            if not skip_missing:
                raise FileNotFoundError(absent[0])
            missing.extend(absent)
            continue
        rn, ta, td = [load(path) for path in paths]
        if any(values.shape != shape for values in [rn, ta, td]):
            raise ValueError(f"Input shape mismatch: {day:%Y%m%d} hour {hour}")
        if not np.isfinite(rn).any():
            continue
        vpd = saturation_vapor_pressure(ta) - saturation_vapor_pressure(td)
        values = (np.maximum(rn, -50.0), ta, savi, vpd)
        for field, array in zip(FIELDS, values):
            valid = np.isfinite(array)
            sums[field] += np.where(valid, array, 0.0)
            counts[field] += valid
    means = {
        field: np.divide(
            sums[field],
            counts[field],
            out=np.full(shape, np.nan),
            where=counts[field] > 0,
        )
        for field in FIELDS
    }
    return means, missing


def infer_shape(root, start, end, tag):
    day = start
    while day < end:
        for hour in range(24):
            path = hourly_paths(root, day, tag, hour)[0]
            if path.exists():
                return np.load(path, mmap_mode="r").shape
        day += timedelta(days=1)
    raise ValueError("No net-radiation inputs were available")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=["e", "w"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--start", required=True, help="First output date YYYYMMDD")
    parser.add_argument("--end", required=True, help="Exclusive output date YYYYMMDD")
    parser.add_argument(
        "--output", type=Path, required=True, help="Moving-average output directory"
    )
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    start = datetime.strptime(args.start, "%Y%m%d")
    end = datetime.strptime(args.end, "%Y%m%d")
    if start >= end or args.window_days < 1:
        parser.error("Require start < end and a positive window length")
    root = args.input_root.expanduser().resolve()
    tag = "fd" + args.region
    first_day = start - timedelta(days=args.window_days)
    raw_end = end - timedelta(days=1)
    shape = infer_shape(root, first_day, raw_end, tag)
    args.output.mkdir(parents=True, exist_ok=True)
    day = start
    while day < end:
        for field in FIELDS:
            path = args.output / f"{day:%Y%m%d}_mean_{tag}_{field}.npy"
            if path.exists():
                raise FileExistsError(path)
        day += timedelta(days=1)

    sums = {field: np.zeros(shape) for field in FIELDS}
    counts = {field: np.zeros(shape, dtype=np.int32) for field in FIELDS}
    history = deque()
    missing = []
    with tempfile.TemporaryDirectory(prefix="topt_daily_", dir=args.output) as scratch:
        day = first_day
        while day < raw_end:
            means, absent = daily_means(root, day, tag, shape, args.skip_missing)
            missing.extend(absent)
            path = Path(scratch) / f"{day:%Y%m%d}.npy"
            daily = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float64, shape=(len(FIELDS),) + shape
            )
            for i, field in enumerate(FIELDS):
                values = means[field]
                daily[i] = values
                valid = np.isfinite(values)
                sums[field] += np.where(valid, values, 0.0)
                counts[field] += valid
            daily.flush()
            del daily, means
            history.append(path)
            if len(history) > args.window_days:
                expired_path = history.popleft()
                expired = np.load(expired_path, mmap_mode="r")
                for i, field in enumerate(FIELDS):
                    valid = np.isfinite(expired[i])
                    sums[field] -= np.where(valid, expired[i], 0.0)
                    counts[field] -= valid
                del expired
                expired_path.unlink()
            label = day + timedelta(days=1)
            if label >= start:
                for field in FIELDS:
                    mean = np.divide(
                        sums[field],
                        counts[field],
                        out=np.full(shape, np.nan),
                        where=counts[field] > 0,
                    )
                    save_scaled(
                        args.output / f"{label:%Y%m%d}_mean_{tag}_{field}.npy", mean
                    )
                print(f"Saved moving averages for {label:%Y%m%d}", flush=True)
            day += timedelta(days=1)
    metadata = dict(
        region=args.region,
        input_root=str(root),
        start=args.start,
        end_exclusive=args.end,
        window_days=args.window_days,
        averaging="equal-weight valid daily means",
        rn_lower_bound_W_m2=-50.0,
        missing_files=missing,
        vpd="e(Ta)-e(Td), using the model saturation-pressure function",
        stored_scale=10000.0,
        nodata=-9999,
    )
    (args.output / "preparation.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
