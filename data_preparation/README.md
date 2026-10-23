# PT-JPL Data Preparation

Prepare optimum-temperature (Topt) fields for regional PT-JPL evapotranspiration estimation. The workflow uses hourly net radiation, air temperature, dew-point temperature, and daily NDVI to calculate moving averages and select Topt for each pixel.

```text
Hourly meteorology + daily NDVI
               ↓
       Moving-average inputs
               ↓
              Topt
```

## Files

```text
.
├── README.md
├── requirements.txt
├── prepare_topt.py
└── calculate_topt.py
```

The scripts can run independently of the ET estimation package.

## Installation

Requires Python 3.9 or later and NumPy. From the repository root:

```bash
cd data_preparation
python -m pip install -r requirements.txt
```

## Input data

Use `--region e` for the east partition or `--region w` for the west partition. `TAG` is `fde` or `fdw`, respectively.

```text
INPUT_ROOT/
└── YEAR/
    ├── r_net_modis001/YYYYMMDDHHMM_input_TAG_rnet.npy
    ├── ta_era001/YYYYMMDDHHMM_input_TAG_ta.npy
    ├── td_era001/YYYYMMDDHHMM_input_TAG_td.npy
    └── ndvi_daily_ahi001/YYYYMMDD0000_daily_TAG_ndvi.npy
```

| Variable | Physical unit |
|---|---|
| Net radiation | W m⁻² |
| Air temperature | °C |
| Dew-point temperature | °C |
| NDVI | Dimensionless |

Source NPY arrays store **physical values × 10000**, with `-9999` as missing data. All arrays must have identical spatial shapes and pixel order. Both 2-D grids and 1-D land-pixel vectors are supported.

## 1. Prepare moving averages

```bash
python prepare_topt.py \
    --region e \
    --input-root /path/to/FD_E_inputs \
    --start 20180115 \
    --end 20190115 \
    --output /path/to/topt_means
```

Dates are UTC, start-inclusive and end-exclusive. Each output date labels the **preceding 14 calendar days**, excluding that date. Hourly inputs are averaged within each day, then valid daily means receive equal weight. Use `--window-days` to change the window length.

For example, the output labelled `20180115` uses January 1–14. The complete example reads hourly data from January 1, 2018 through January 13, 2019. The input archive must cover those dates. Add `--skip-missing` to omit unavailable files.

The output directory contains four arrays for each date:

```text
YYYYMMDD_mean_TAG_r_net.npy
YYYYMMDD_mean_TAG_ta.npy
YYYYMMDD_mean_TAG_savi.npy
YYYYMMDD_mean_TAG_vpd.npy
```

SAVI is calculated as `0.45 * NDVI + 0.132`. VPD is calculated as `e(Ta) - e(Td)` in kPa, using `e(T) = 0.61121 * exp(17.502 * T / (T + 240.97))`. The arrays store physical values ×10000 as `int32`, with `-9999` for missing data.

## 2. Calculate Topt

```bash
python calculate_topt.py \
    --region e \
    --input-root /path/to/topt_means \
    --start 20180115 \
    --end 20190115 \
    --output /path/to/topt.npy
```

For each pixel, Topt is the mean air temperature at the date maximizing:

```text
Rn_mean * Ta_mean * SAVI_mean / max(VPD_mean, 0.5)
```

At least 11 dates with all four quantities valid are required by default. Use `--min-valid-days` to change the requirement. Pixels with insufficient valid observations remain missing. `--chunk-size` controls the number of pixels processed per block.

Existing moving-average files can be used directly with this step if their units, averaging convention, spatial shape, and pixel order match. Add `--skip-missing` to skip dates without a complete set of four files.

## Output and ET use

The output is an `int32` NPY array storing **Topt in °C ×10000**, with `-9999` as missing data. A JSON sidecar records the input period and calculation settings. Existing output files are not overwritten.

Pass the generated file explicitly to the ET estimation program:

```bash
python ../run_spatial.py \
    --year 2018 \
    --region e \
    --input-root /path/to/FD_E_inputs \
    --topt /path/to/topt.npy \
    --output /path/to/et_output
```

Use `python prepare_topt.py --help` and `python calculate_topt.py --help` to view the available parameters.
