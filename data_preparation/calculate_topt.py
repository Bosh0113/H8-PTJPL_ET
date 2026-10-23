"""Calculate spatial optimum temperature from moving-average inputs."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FIELDS = ("r_net", "ta", "savi", "vpd")
SCALE = 10000.0


def decode(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values == -9999, np.nan, values / SCALE)


def save_scaled(path, values):
    valid = np.isfinite(values)
    scaled = np.rint(values[valid] * SCALE)
    limits = np.iinfo(np.int32)
    if np.any((scaled < limits.min) | (scaled > limits.max)):
        raise ValueError(f"Values exceed int32 range: {path}")
    output = np.full(values.shape, -9999, dtype=np.int32)
    output[valid] = scaled.astype(np.int32)
    np.save(path, output)


def update_selection(best_score, best_temperature, valid_count, rn, ta, savi, vpd):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        score = rn * ta * savi / np.maximum(vpd, 0.5)
    valid = (
        np.isfinite(rn)
        & np.isfinite(ta)
        & np.isfinite(savi)
        & np.isfinite(vpd)
        & np.isfinite(score)
    )
    valid_count += valid
    better = valid & (score > best_score)
    best_score[better] = score[better]
    best_temperature[better] = ta[better]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=["e", "w"], required=True)
    parser.add_argument(
        "--input-root", type=Path, required=True, help="Moving-average input directory"
    )
    parser.add_argument("--start", required=True, help="Inclusive date YYYYMMDD")
    parser.add_argument("--end", required=True, help="Exclusive date YYYYMMDD")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output Topt NPY file"
    )
    parser.add_argument("--min-valid-days", type=int, default=11)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    start = datetime.strptime(args.start, "%Y%m%d")
    end = datetime.strptime(args.end, "%Y%m%d")
    if start >= end or args.chunk_size < 1:
        parser.error("Require start < end and a positive chunk size")
    if not 1 <= args.min_valid_days <= (end - start).days:
        parser.error("--min-valid-days must fit within the date interval")
    if args.output.suffix != ".npy":
        parser.error("--output must have a .npy extension")
    if args.output.exists():
        raise FileExistsError(args.output)

    root = args.input_root.expanduser().resolve()
    tag = "fd" + args.region
    best_score = best_temperature = valid_count = None
    shape = None
    processed = 0
    missing = []
    day = start
    while day < end:
        paths = [root / f"{day:%Y%m%d}_mean_{tag}_{field}.npy" for field in FIELDS]
        absent = [str(path) for path in paths if not path.exists()]
        if absent:
            if not args.skip_missing:
                raise FileNotFoundError(absent[0])
            missing.extend(absent)
            day += timedelta(days=1)
            continue
        arrays = [np.load(path, mmap_mode="r") for path in paths]
        if shape is None:
            shape = arrays[0].shape
            best_score = np.full(shape, -np.inf)
            best_temperature = np.full(shape, np.nan)
            valid_count = np.zeros(shape, dtype=np.int32)
        if any(array.shape != shape for array in arrays):
            raise ValueError(f"Input shape mismatch on {day:%Y%m%d}")
        flat = [array.reshape(-1) for array in arrays]
        for offset in range(0, best_score.size, args.chunk_size):
            part = slice(offset, offset + args.chunk_size)
            update_selection(
                best_score.reshape(-1)[part],
                best_temperature.reshape(-1)[part],
                valid_count.reshape(-1)[part],
                *(decode(array[part]) for array in flat),
            )
        processed += 1
        if processed % 30 == 0:
            print(f"Processed {processed} dates through {day:%Y%m%d}", flush=True)
        day += timedelta(days=1)

    if shape is None:
        raise ValueError("No moving-average inputs were available")
    best_temperature[valid_count < args.min_valid_days] = np.nan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_scaled(args.output, best_temperature)
    metadata = dict(
        region=args.region,
        input_root=str(root),
        start=args.start,
        end_exclusive=args.end,
        min_valid_days=args.min_valid_days,
        processed_dates=processed,
        missing_files=missing,
        selection_score="Rn_mean * Ta_mean * SAVI_mean / max(VPD_mean, 0.5)",
        units="degrees C",
        stored_scale=SCALE,
        nodata=-9999,
        valid_pixels=int(np.isfinite(best_temperature).sum()),
    )
    Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {args.output}; valid pixels: {metadata['valid_pixels']}", flush=True)


if __name__ == "__main__":
    main()
