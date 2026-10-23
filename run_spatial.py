"""Run hourly spatial PT-JPL estimation."""

import argparse
import calendar
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ptjpl_spatial import MODEL_VERSION, fapar_from_evi2, ptjpl_spatial

SCALE = 10000.0


def load(path):
    array = np.load(path).astype(np.float64)
    array[array == -9999] = np.nan
    return array / SCALE


def merge_lst(ahi_c, era_c=None):
    """Select AHI LST with optional ERA5 filling."""
    ahi_c = np.asarray(ahi_c)
    use_ahi = np.isfinite(ahi_c) & (ahi_c != 0)
    if era_c is None:
        merged = np.where(use_ahi, ahi_c, np.nan)
        era_pixels = 0
    else:
        era_c = np.asarray(era_c)
        if ahi_c.shape != era_c.shape:
            raise ValueError("AHI and ERA5 LST shapes do not match")
        merged = np.where(use_ahi, ahi_c, era_c)
        era_pixels = int((~use_ahi & np.isfinite(era_c)).sum())
    merged = np.where(np.isfinite(merged), merged, np.nan)
    counts = dict(
        lst_ahi_pixels=int(use_ahi.sum()),
        lst_era_pixels=era_pixels,
        lst_missing_pixels=int((~np.isfinite(merged)).sum()),
    )
    return merged, counts


def estimate_chunks(inputs, chunk_size, quality_counts=None):
    """Calculate fluxes in spatial blocks."""
    shape = inputs[0].shape
    if any(array.shape != shape for array in inputs):
        raise ValueError("Input shapes do not match")
    flat = [array.reshape(-1) for array in inputs]
    fields = ["et_a", "et_c", "et_i", "et_s", "et_p"]
    out = {k: np.empty(shape, dtype=np.float32) for k in fields}
    valid = capped = 0
    for start in range(0, flat[0].size, chunk_size):
        part = slice(start, start + chunk_size)
        result = ptjpl_spatial(*(array[part] for array in flat))
        for k in fields:
            out[k].reshape(-1)[part] = np.where(
                np.isfinite(result[k]), result[k] * SCALE, -9999.0
            )
        valid += int(np.isfinite(result["et_a"]).sum())
        capped += int(np.sum(result["closure_residual"] > 1e-6))
        if quality_counts is not None:
            for key in fields:
                name = "valid_" + key
                quality_counts[name] = quality_counts.get(name, 0) + int(
                    np.isfinite(result[key]).sum()
                )
            for key, values in result.items():
                if key.startswith("qc_"):
                    quality_counts[key] = quality_counts.get(key, 0) + int(
                        np.count_nonzero(values)
                    )
    return out, valid, capped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        choices=["e", "w"],
        required=True,
        help="Domain partition: e (east) or w (west)",
    )
    parser.add_argument("--year", type=int, required=True, help="Processing year")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Domain input directory containing year subdirectories",
    )
    parser.add_argument(
        "--output", type=Path, help="Default: this code directory/results_REGION_YEAR"
    )
    parser.add_argument(
        "--topt",
        type=Path,
        required=True,
        help="Optimum-temperature NPY selected for this year and domain",
    )
    parser.add_argument("--start", help="Inclusive UTC YYYYMMDDHHMM")
    parser.add_argument("--end", help="Exclusive UTC YYYYMMDDHHMM")
    parser.add_argument(
        "--skip-missing", action="store_true", help="Log missing inputs and continue"
    )
    parser.add_argument(
        "--fill-lst-era5",
        action="store_true",
        help="Fill zero/missing AHI LST with ERA5; default: no filling",
    )
    parser.add_argument(
        "--fapar-max",
        type=Path,
        help="Existing physical-unit NPY maximum, no DN scaling",
    )
    parser.add_argument("--chunk-size", type=int, default=250000)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent / f"results_{args.region}_{args.year}"
        )
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    configured_root = args.input_root
    root = configured_root.expanduser().resolve()
    tag = "fd" + args.region
    year_dir = root / str(args.year)
    topt_path = args.topt.expanduser().resolve()
    topt = load(topt_path)
    start = (
        datetime.strptime(args.start, "%Y%m%d%H%M")
        if args.start
        else datetime(args.year, 1, 1)
    )
    end = (
        datetime.strptime(args.end, "%Y%m%d%H%M")
        if args.end
        else datetime(args.year + 1, 1, 1)
    )
    if not datetime(args.year, 1, 1) <= start < end <= datetime(args.year + 1, 1, 1):
        parser.error("Require start < end within the specified year")
    if start.minute or end.minute:
        parser.error("Start/end must be aligned to whole hours")
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "model_version.json"
    expected = dict(
        model_version=MODEL_VERSION,
        region=args.region,
        year=args.year,
        input_root=str(root),
        topt=str(topt_path),
        fill_lst_era5=args.fill_lst_era5,
    )
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError(
                "Output directory model/input configuration differs; use a new directory"
            )
    else:
        if any(args.output.glob("*_output_*.npz")):
            raise ValueError(
                "Unversioned existing ET outputs: use a new output directory"
            )
        marker.write_text(json.dumps(expected, indent=2))
    if args.fapar_max:
        fmax = np.load(args.fapar_max)
        fmax_source = str(args.fapar_max)
        if fmax.dtype != np.dtype("float64"):
            raise ValueError(
                "fAPARmax must be float64; regenerate the cache using the annual calculation"
            )
        cache_meta_path = Path(str(args.fapar_max) + ".json")
        if cache_meta_path.exists():
            cache_meta = json.loads(cache_meta_path.read_text())
            if any(
                cache_meta.get(k) != expected[k]
                for k in ["model_version", "region", "year", "input_root"]
            ):
                raise ValueError("fAPARmax cache metadata does not match this run")
            missing_days = cache_meta.get("missing_evi_days")
        else:
            missing_days = None
    else:
        cache = args.output / f"{args.year}_{tag}_fapar_max_float64.npy"
        if cache.exists():
            raise FileExistsError(
                f"{cache}: reuse explicitly via --fapar-max, or remove to rebuild"
            )
        fmax = np.full(topt.shape, np.nan, dtype=np.float64)
        missing_days = []
        days_in_year = 366 if calendar.isleap(args.year) else 365
        for n in range(days_in_year):
            if n % 30 == 0:
                print(
                    f"Building float64 fAPARmax: {n}/{days_in_year} days ({tag}, {args.year})",
                    flush=True,
                )
            day = datetime(args.year, 1, 1) + timedelta(days=n)
            path = (
                year_dir
                / "evi2_daily_ahi001"
                / f"{day:%Y%m%d}0000_daily_{tag}_evi2.npy"
            )
            if not path.exists():
                if not args.skip_missing:
                    raise FileNotFoundError(path)
                missing_days.append(str(path))
                continue
            evi = load(path)
            if evi.shape != topt.shape:
                raise ValueError(f"Shape mismatch: {path}")
            np.fmax(fmax, fapar_from_evi2(evi), out=fmax)
        if not np.any(np.isfinite(fmax)):
            raise ValueError("No valid annual EVI2 observations")
        np.save(cache, fmax)
        Path(str(cache) + ".json").write_text(
            json.dumps(
                dict(
                    expected,
                    missing_evi_days=missing_days,
                    dtype="float64",
                    fapar_formula="(1.3632*EVI2-0.048)*0.45+0.15",
                ),
                indent=2,
            )
        )
        fmax_source = str(cache)
    if fmax.shape != topt.shape:
        raise ValueError("fAPAR maximum shape does not match Topt")
    metadata = dict(
        year=args.year,
        region=args.region,
        input_root=str(root),
        configured_input_root=str(configured_root),
        topt=str(topt_path),
        fill_lst_era5=args.fill_lst_era5,
        lst_source=(
            "finite nonzero AHI, otherwise ERA5"
            if args.fill_lst_era5
            else "finite nonzero AHI only; no filling"
        ),
        fapar_max=fmax_source,
        missing_evi_days=missing_days,
        units="W m-2",
        stored_scale=SCALE,
        nodata=-9999,
        start=start.isoformat(),
        end_exclusive=end.isoformat(),
        model=MODEL_VERSION,
        rh_formula="same es(Ta) and es(Td); tolerance 1e-6",
        constraints="fg clipped; fM roundoff tolerance 1e-6; component-specific validity",
    )
    run_id = f"{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"
    (args.output / f"run_{run_id}.json").write_text(json.dumps(metadata, indent=2))
    with (args.output / f"status_{run_id}.jsonl").open("a") as log:
        now = start
        while now < end:
            dest = args.output / f"{now:%Y%m%d%H%M}_output_{tag}_et.npz"
            if dest.exists():
                raise FileExistsError(dest)
            paths = [
                year_dir / d / f"{now:%Y%m%d%H%M}_input_{tag}_{v}.npy"
                for d, v in [
                    ("r_net_modis001", "rnet"),
                    ("td_era001", "td"),
                    ("ta_era001", "ta"),
                ]
            ]
            paths += [
                year_dir / d / f"{now:%Y%m%d}0000_daily_{tag}_{v}.npy"
                for d, v in [
                    ("ndvi_daily_ahi001", "ndvi"),
                    ("evi2_daily_ahi001", "evi2"),
                ]
            ]
            paths += [year_dir / "lst_ahi001" / f"{now:%Y%m%d%H%M}_input_{tag}_lst.npy"]
            try:
                rn, td, ta, ndvi, evi, ahi_lst = [load(path) for path in paths]
                era_lst = (
                    load(
                        year_dir
                        / "lst_era001"
                        / f"{now:%Y%m%d%H%M}_input_{tag}_lst.npy"
                    )
                    if args.fill_lst_era5
                    else None
                )
                lst, lst_counts = merge_lst(ahi_lst, era_lst)
                quality_counts = {}
                fluxes, valid, capped = estimate_chunks(
                    [rn, td, ta, ndvi, evi, topt, lst, fmax],
                    args.chunk_size,
                    quality_counts,
                )
                np.savez_compressed(dest, **fluxes)
                status = dict(
                    time=f"{now:%Y%m%d%H%M}",
                    status="ok",
                    valid_et=valid,
                    capped_pixels=capped,
                    **lst_counts,
                    **quality_counts,
                )
            except FileNotFoundError as exc:
                if not args.skip_missing:
                    raise
                status = dict(
                    time=f"{now:%Y%m%d%H%M}", status="missing", error=str(exc)
                )
            log.write(json.dumps(status) + "\n")
            log.flush()
            print(status, flush=True)
            now += timedelta(hours=1)


if __name__ == "__main__":
    main()
