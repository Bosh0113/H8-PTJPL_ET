"""Spatial PT-JPL model. Temperatures in degrees C; fluxes in W m-2."""

import numpy as np

MODEL_VERSION = "ptjpl-spatial"
RATIO_TOLERANCE = 1e-6


def bounded(x, lo=0.0, hi=1.0):
    return np.where(np.isfinite(x) & (x >= lo) & (x <= hi), x, np.nan)


def fapar_from_evi2(evi2):
    evi2 = np.asarray(evi2, dtype=np.float64)
    return bounded((1.3632 * evi2 - 0.048) * 0.45 + 0.15)


def saturation_vapor_pressure(t_c):
    t_c = np.asarray(t_c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pressure = 0.61121 * np.exp(17.502 * t_c / (t_c + 240.97))
    return np.where(
        np.isfinite(t_c) & (t_c > -240.97) & np.isfinite(pressure) & (pressure > 0),
        pressure,
        np.nan,
    )


def ptjpl_spatial(rn, td_c, ta_c, ndvi, evi2, topt_c, lst_c, fapar_max):
    """Calculate fluxes from arrays with identical spatial shapes."""
    inputs = [
        np.asarray(x, dtype=float)
        for x in (rn, td_c, ta_c, ndvi, evi2, topt_c, lst_c, fapar_max)
    ]
    if len({x.shape for x in inputs}) != 1:
        raise ValueError("All inputs must have identical spatial shapes")
    rn, td, ta, ndvi, evi2, topt, lst, fapar_max = inputs

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        es = saturation_vapor_pressure(ta)
        ea = saturation_vapor_pressure(td)
        rh_raw = ea / es
        rh = np.where(
            np.isfinite(rh_raw) & (rh_raw >= 0) & (rh_raw <= 1 + RATIO_TOLERANCE),
            np.minimum(rh_raw, 1.0),
            np.nan,
        )
        vpd = np.where(
            np.isfinite(rh), np.maximum(np.maximum(es, 1.0) - ea, 0.0), np.nan
        )
        delta = 240.97 * 17.502 * es / (ta + 240.97) ** 2
        epsilon = delta / (delta + 0.0662)
        wet = np.where(np.isfinite(es) & (ta <= 0), 0.0, rh**4)
        fsm = np.where(np.isfinite(es) & (ta < 0), 0.0, bounded(rh**vpd))

        ndvi_ok = np.isfinite(ndvi) & (ndvi >= 0) & (ndvi <= 1)
        fipar = np.where(ndvi_ok, np.clip(ndvi - 0.05, 0.0, 0.95), np.nan)
        zero_cover = ndvi_ok & (fipar == 0)
        lai = -np.log1p(-fipar) / 0.5
        rn_s = rn * np.exp(-0.6 * lai)
        rn_c = rn - rn_s
        fapar = fapar_from_evi2(evi2)
        fg_raw = fapar / fipar
        fg = np.where(
            zero_cover,
            0.0,
            np.where(np.isfinite(fg_raw), np.clip(fg_raw, 0.0, 1.0), np.nan),
        )
        fm_raw = np.where(
            np.isfinite(fapar_max) & (fapar_max > 0), fapar / fapar_max, np.nan
        )
        fm = np.where(
            np.isfinite(fm_raw) & (fm_raw >= 0) & (fm_raw <= 1 + RATIO_TOLERANCE),
            np.minimum(fm_raw, 1.0),
            np.nan,
        )
        ft_warm = np.where(
            np.isfinite(topt) & (topt != 0),
            np.exp(-(((ta - topt) / topt) ** 2)),
            np.nan,
        )
        ft = np.where(np.isfinite(es) & (ta < -5), 0.05, ft_warm)

        lai_g = -np.log((0.69 - (0.45 * ndvi + 0.132)) / 0.59) / 0.91
        g_high = rn * (0.05 + 0.18 * np.exp(-0.521 * lai_g))
        g_low = 1.8 * lst + 0.084 * rn
        g = np.where(lai >= 0.5, g_high, g_low)
        g = np.minimum(np.maximum(g, 0.0), 0.35 * rn_s)
        g_ok = ndvi_ok & np.isfinite(rn) & np.isfinite(g)
        g = np.where(g_ok, g, np.nan)

        soil_available = rn_s - g
        le_s = (wet + fsm * (1 - wet)) * 1.26 * epsilon * soil_available

        le_s = np.where(
            (ta < 0) & np.isfinite(epsilon) & g_ok & (soil_available >= 0), 0.0, le_s
        )
        le_s = np.where(g_ok & np.isfinite(le_s) & (le_s >= 0), le_s, np.nan)

        le_c = 1.26 * (1 - wet) * fg * ft * fm * epsilon * rn_c
        canopy_ok = ndvi_ok & np.isfinite(rn_c)

        zero_t = canopy_ok & (zero_cover | ((wet == 1) & np.isfinite(epsilon)))
        le_c = np.where(zero_t, 0.0, le_c)
        le_c = np.where(canopy_ok & np.isfinite(le_c), np.maximum(le_c, 0.0), np.nan)

        le_i = wet * 1.26 * epsilon * rn_c
        le_i = np.where(canopy_ok & zero_cover, 0.0, le_i)
        le_i = np.where(canopy_ok & np.isfinite(le_i), np.maximum(le_i, 0.0), np.nan)

        le_sum = le_s + le_c + le_i
        le = np.minimum(le_sum, rn)
        le = np.where(np.isfinite(le) & (le >= 0), le, np.nan)
        pet = 1.26 * epsilon * (rn - g)

    result = dict(
        et_a=le,
        et_c=le_c,
        et_i=le_i,
        et_s=le_s,
        et_p=pet,
        soil_heat_flux=g,
        fapar=fapar,
        f_m=fm,
        f_g=fg,
        rh=rh,
        vpd=vpd,
        closure_residual=le_sum - le,
    )
    result = {k: np.where(np.isfinite(v), v, np.nan) for k, v in result.items()}
    result.update(
        qc_fg_clipped=ndvi_ok & (fipar > 0) & np.isfinite(fg_raw) & (fg_raw > 1),
        qc_fm_roundoff=(
            np.isfinite(fm_raw) & (fm_raw > 1) & (fm_raw <= 1 + RATIO_TOLERANCE)
        ),
        qc_rh_roundoff=(
            np.isfinite(rh_raw) & (rh_raw > 1) & (rh_raw <= 1 + RATIO_TOLERANCE)
        ),
        qc_zero_cover=zero_cover,
    )
    return result
