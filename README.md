# Himawari-8 Regional Instantaneous ET

PT-JPL-based estimation of regional instantaneous evapotranspiration (ET) within the Himawari-8 observation domain. The main inputs are net radiation, air temperature, dew-point temperature, vegetation indices (NDVI and EVI2), and land surface temperature (LST).

The workflow processes hourly timestamps and partitions ET into canopy transpiration, interception evaporation, and soil evaporation. Outputs are expressed as latent heat flux in **W m⁻²**, not water depth in mm.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── ptjpl_spatial.py
├── run_spatial.py
└── data/
    └── land_index/
        ├── ahi_land_E_index.npz
        └── ahi_land_W_index.npz
```

`run_spatial.py` selects the year and domain through command-line arguments and handles input and output. It calls the shared model in `ptjpl_spatial.py`. The land indexes are included; meteorological and vegetation input arrays must be prepared separately.

## Installation

Requires Python 3.9 or later and NumPy. Run from the repository directory:

```bash
python -m pip install -r requirements.txt
```

## Running the model

Use `--year` to select the processing year and `--region e` or `--region w` to select the east or west partition. `--input-root` points to the input directory for that partition.

```bash
python run_spatial.py --year 2018 --region e --input-root /path/to/FD_E_inputs
python run_spatial.py --year 2019 --region e --input-root /path/to/FD_E_inputs
python run_spatial.py --year 2018 --region w --input-root /path/to/FD_W_inputs
python run_spatial.py --year 2019 --region w --input-root /path/to/FD_W_inputs
```

The year is not limited to 2018 and 2019; other years use the same input layout.

To process one hour:

```bash
python run_spatial.py --year 2018 --region e \
    --input-root /path/to/FD_E_inputs \
    --start 201801010300 \
    --end 201801010400 \
    --output /path/to/output
```

Times are UTC, on whole hours. The start is inclusive and the end is exclusive. Without a time range, the selected calendar year is processed.

LST uses AHI only by default. Add `--fill-lst-era5` to fill missing or zero AHI LST with ERA5. ERA5 LST files are required only when this option is enabled. Use a separate output directory for each LST mode.

```bash
python run_spatial.py --year 2018 --region e \
    --input-root /path/to/FD_E_inputs \
    --fill-lst-era5 \
    --output /path/to/output_with_filling
```

Use `python run_spatial.py --help` to list all parameters.

## Input data

`--input-root` points to one domain directory. Use `TAG=fde` for the east partition and `TAG=fdw` for the west partition.

```text
INPUT_ROOT/
├── 2019_t_opt.npy
└── YEAR/
    ├── r_net_modis001/YYYYMMDDHHMM_input_TAG_rnet.npy
    ├── ta_era001/YYYYMMDDHHMM_input_TAG_ta.npy
    ├── td_era001/YYYYMMDDHHMM_input_TAG_td.npy
    ├── lst_ahi001/YYYYMMDDHHMM_input_TAG_lst.npy
    ├── lst_era001/YYYYMMDDHHMM_input_TAG_lst.npy  [optional]
    ├── ndvi_daily_ahi001/YYYYMMDD0000_daily_TAG_ndvi.npy
    └── evi2_daily_ahi001/YYYYMMDD0000_daily_TAG_evi2.npy
```

Replace `YEAR`, `TAG`, and timestamps with actual values.

| Variable | Description | Physical unit |
|---|---|---|
| Rn | Net radiation | W m⁻² |
| Ta | Air temperature | °C |
| Td | Dew-point temperature | °C |
| NDVI | Normalized Difference Vegetation Index | Dimensionless |
| EVI2 | Two-band Enhanced Vegetation Index | Dimensionless |
| LST | Land surface temperature | °C |
| Topt | Optimum temperature field | °C |

Source NPY arrays store **physical values × 10000**, with `-9999` as missing data. Arrays must have identical shapes and pixel order. Both two-dimensional grids and one-dimensional land-pixel vectors are supported.

The default Topt filename is `2019_t_opt.npy`, independent of `--year`. Use `--topt /path/to/topt.npy` to select the appropriate temperature field for another study or reference period.

### Annual fAPAR maximum

The program calculates each pixel's annual maximum fAPAR from daily EVI2. Daily EVI2 for the selected year is therefore needed even for a short run, unless a compatible cache is supplied:

```bash
--fapar-max /path/to/YEAR_TAG_fapar_max_float64.npy
```

The generated cache is accompanied by a `.npy.json` metadata file. It stores physical values as `float64`, **without** the 10000 scaling. `--skip-missing` allows missing daily or hourly input files to be skipped.

## Outputs

The default output directory beside the code is `results_REGION_YEAR`, for example `results_e_2018` or `results_w_2019`. Use `--output` to select another directory. Existing hourly files are not overwritten.

Each hour produces `YYYYMMDDHHMM_output_TAG_et.npz`:

| Field | Description |
|---|---|
| `et_a` | Total latent heat flux |
| `et_c` | Canopy transpiration |
| `et_i` | Interception evaporation |
| `et_s` | Soil evaporation |
| `et_p` | Potential latent heat flux |

Arrays retain the input shape and are stored as `float32`. Mask `-9999` before dividing by 10000 to obtain W m⁻²:

```python
import numpy as np

with np.load("201801010300_output_fde_et.npz") as output:
    dn = output["et_a"].astype(np.float64)

et = np.where(dn == -9999, np.nan, dn / 10000.0)
```

## Spatial grid

The supplied land indexes refer to a regular **0.01° × 0.01° latitude–longitude grid**, rather than the native AHI scanning grid.

| Domain | Longitude extent | Latitude extent | Grid shape (rows × columns) |
|---|---|---|---|
| Full domain | 85°E–205°E (155°W), crossing 180° | 60°S–60°N | 12000 × 12000 |
| East partition | 85°E–180° | 60°S–60°N | 12000 × 9500 |
| West partition | 180°–205°E (155°W) | 60°S–60°N | 12000 × 2500 |

Rows run from north to south; full-grid columns run from west to east. Only the indexed land pixels are stored in one-dimensional outputs, and valid ET coverage varies with the availability of the required inputs.

## Land indexes and full-domain reconstruction

The files in `data/land_index/` contain zero-based `rows` and `cols` arrays for the land-pixel vectors:

| Partition | Index file | Partition grid shape | Land pixels |
|---|---|---|---:|
| East (`fde`) | `ahi_land_E_index.npz` | 12000 × 9500 | 29,147,644 |
| West (`fdw`) | `ahi_land_W_index.npz` | 12000 × 2500 | 204,995 |

The two partitions form a **12000 × 12000** full-domain grid. West-partition columns are offset by 9500 when combining the grids. These indexes apply to the supplied grid convention and vector order; inputs prepared on another grid require corresponding indexes.

The following example restores total ET at one timestamp without spatial averaging. Run it from the repository directory after generating both domain outputs:

```python
from pathlib import Path

import numpy as np

stamp = "201801010300"
full_grid = np.full((12000, 12000), np.nan, dtype=np.float32)

for domain, tag, column_offset in [("E", "fde", 0), ("W", "fdw", 9500)]:
    index_path = Path("data/land_index") / f"ahi_land_{domain}_index.npz"
    with np.load(index_path) as index:
        rows = index["rows"]
        cols = index["cols"]

    output_path = (
        Path(f"results_{domain.lower()}_2018")
        / f"{stamp}_output_{tag}_et.npz"
    )
    with np.load(output_path) as output:
        dn = output["et_a"]

    values = np.where(dn == -9999, np.nan, dn / 10000.0)
    full_grid[rows, cols + column_offset] = values

np.save(f"{stamp}_full_domain_et.npy", full_grid)
```
