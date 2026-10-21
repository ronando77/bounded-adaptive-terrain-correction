"""Production-visible local terrain geometry features without reference targets."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    radius_m: float = 20.0
    sector_count: int = 8
    fit_rmse_floor_m: float = 0.08
    fit_rmse_relief_fraction: float = 0.18
    step_min_jump_m: float = 0.25
    step_improvement_ratio: float = 0.72
    step_orientation_increment_deg: float = 5.0
    step_min_side_fraction: float = 0.15
    step_min_distance_m: float = 1.0
    cone_sector_slope_mad_fraction: float = 0.25
    minimum_filled_coverage: float = 0.98
    minimum_raw_coverage: float = 0.98


def huber_lstsq(
    design: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Small deterministic iteratively reweighted Huber fit."""

    good = np.all(np.isfinite(design), axis=1) & np.isfinite(response)
    if np.count_nonzero(good) < design.shape[1] + 3:
        return np.full(design.shape[1], np.nan), math.inf
    matrix = design[good]
    values = response[good]
    weights = np.ones(len(values), dtype=float)
    beta = np.zeros(matrix.shape[1], dtype=float)
    for _ in range(8):
        root = np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(
            matrix * root[:, None],
            values * root,
            rcond=None,
        )
        residual = values - matrix @ beta
        scale = 1.4826 * np.median(
            np.abs(residual - np.median(residual))
        ) + 1e-8
        scaled = np.abs(residual) / (1.345 * scale)
        weights = np.ones_like(scaled)
        large = scaled > 1.0
        weights[large] = 1.0 / scaled[large]
    return beta, float(np.sqrt(np.mean((values - matrix @ beta) ** 2)))


def best_straight_step(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: GeometryConfig,
) -> dict[str, float]:
    """Find a two-level straight boundary using geometry only."""

    count = len(z)
    minimum = max(8, int(math.ceil(config.step_min_side_fraction * count)))
    best = {
        "step_rmse_m": math.inf,
        "step_jump_m": math.nan,
        "step_distance_m": math.nan,
        "step_orientation_deg": math.nan,
        "step_station_side_offset_m": math.nan,
    }
    if count < 2 * minimum + 1:
        return best
    for angle_deg in np.arange(0.0, 180.0, config.step_orientation_increment_deg):
        angle = math.radians(float(angle_deg))
        projection = x * math.cos(angle) + y * math.sin(angle)
        order = np.argsort(projection)
        projected = projection[order]
        values = z[order]
        cumulative = np.cumsum(values)
        cumulative2 = np.cumsum(values * values)
        indices = np.arange(minimum, count - minimum + 1)
        valid = projected[indices - 1] < projected[indices]
        boundary = (projected[indices - 1] + projected[indices]) / 2.0
        valid &= np.abs(boundary) >= config.step_min_distance_m
        valid &= np.abs(boundary) < config.radius_m - 1.0
        if not np.any(valid):
            continue
        indices = indices[valid]
        boundary = boundary[valid]
        left_sum = cumulative[indices - 1]
        left_sum2 = cumulative2[indices - 1]
        right_sum = cumulative[-1] - left_sum
        right_sum2 = cumulative2[-1] - left_sum2
        left_count = indices.astype(float)
        right_count = count - left_count
        left_mean = left_sum / left_count
        right_mean = right_sum / right_count
        left_sse = np.maximum(0.0, left_sum2 - left_sum * left_sum / left_count)
        right_sse = np.maximum(
            0.0,
            right_sum2 - right_sum * right_sum / right_count,
        )
        rmse = np.sqrt((left_sse + right_sse) / count)
        local = int(np.argmin(rmse))
        if float(rmse[local]) >= best["step_rmse_m"]:
            continue
        threshold = float(boundary[local])
        station_offset = (
            float(left_mean[local])
            if 0.0 < threshold
            else float(right_mean[local])
        )
        best = {
            "step_rmse_m": float(rmse[local]),
            "step_jump_m": float(abs(right_mean[local] - left_mean[local])),
            "step_distance_m": float(abs(threshold)),
            "step_orientation_deg": float(angle_deg),
            "step_station_side_offset_m": station_offset,
        }
    return best


def audit_local_geometry(
    dx: np.ndarray,
    dy: np.ndarray,
    relative_height_m: np.ndarray,
    raw_valid: np.ndarray,
    cell_size_m: float,
    config: GeometryConfig = GeometryConfig(),
) -> dict[str, float | int | bool | str]:
    """Extract deployable local features and preliminary physical legality flags."""

    radius = np.hypot(dx, dy)
    circle = radius <= config.radius_m
    filled = circle & np.isfinite(relative_height_m)
    expected = int(np.count_nonzero(circle))
    count = int(np.count_nonzero(filled))
    raw_count = int(np.count_nonzero(circle & raw_valid))
    coverage = count / max(expected, 1)
    raw_coverage = raw_count / max(expected, 1)
    if count < 20:
        return {
            "cell_count": count,
            "filled_coverage": coverage,
            "raw_coverage": raw_coverage,
            "geometry_status": "insufficient_cells",
            "support_ok": False,
        }
    x = np.asarray(dx, dtype=float)[filled]
    y = np.asarray(dy, dtype=float)[filled]
    radial = radius[filled]
    z = np.asarray(relative_height_m, dtype=float)[filled]
    relief = float(np.quantile(z, 0.95) - np.quantile(z, 0.05))
    limit = max(
        config.fit_rmse_floor_m,
        config.fit_rmse_relief_fraction * max(relief, 0.1),
    )
    plane_beta, plane_rmse = huber_lstsq(
        np.column_stack([x, y, np.ones_like(x)]),
        z,
    )
    cone_beta, cone_rmse = huber_lstsq(
        np.column_stack([radial, np.ones_like(radial)]),
        z,
    )
    sector_angle = 2.0 * math.pi / config.sector_count
    azimuth = np.mod(np.arctan2(y, x), 2.0 * math.pi)
    sector_slopes = []
    sector_constant_rmse = []
    sector_constant_accept = 0
    for index in range(config.sector_count):
        use = (
            (azimuth >= index * sector_angle)
            & (azimuth < (index + 1) * sector_angle)
        )
        if np.count_nonzero(use) < 8:
            sector_slopes.append(math.nan)
            sector_constant_rmse.append(math.inf)
            continue
        radial_beta, _ = huber_lstsq(
            np.column_stack([radial[use], np.ones(np.count_nonzero(use))]),
            z[use],
        )
        sector_slopes.append(float(radial_beta[0]))
        residual = z[use] - np.median(z[use])
        rmse = float(np.sqrt(np.mean(residual**2)))
        sector_constant_rmse.append(rmse)
        sector_constant_accept += int(rmse <= limit)
    slopes = np.asarray(sector_slopes, dtype=float)
    finite_slopes = slopes[np.isfinite(slopes)]
    slope_median = (
        float(np.median(finite_slopes)) if len(finite_slopes) else math.nan
    )
    slope_mad = (
        float(np.median(np.abs(finite_slopes - slope_median)))
        if len(finite_slopes)
        else math.inf
    )
    same_sign_fraction = (
        float(np.mean(np.sign(finite_slopes) == np.sign(slope_median)))
        if len(finite_slopes) and slope_median != 0
        else 0.0
    )
    step = best_straight_step(x, y, z, config)
    station_offset_limit = max(0.15, 0.10 * max(relief, 0.1))
    support_ok = (
        coverage >= config.minimum_filled_coverage
        and raw_coverage >= config.minimum_raw_coverage
    )
    cone_candidate = bool(
        support_ok
        and cone_rmse <= limit
        and same_sign_fraction >= 0.875
        and slope_mad
        <= config.cone_sector_slope_mad_fraction
        * max(abs(slope_median), 1e-3)
        and abs(float(cone_beta[1])) <= station_offset_limit
    )
    plane_candidate = bool(
        support_ok
        and plane_rmse <= limit
        and abs(float(plane_beta[2])) <= station_offset_limit
    )
    step_candidate = bool(
        support_ok
        and step["step_jump_m"] >= config.step_min_jump_m
        and step["step_rmse_m"] <= limit
        and step["step_rmse_m"]
        <= config.step_improvement_ratio * max(plane_rmse, 1e-8)
        and abs(step["step_station_side_offset_m"]) <= station_offset_limit
    )
    surface = np.asarray(relative_height_m, dtype=float)
    gradient_y, gradient_x = np.gradient(surface, cell_size_m, cell_size_m)
    slope_magnitude = np.hypot(gradient_x, gradient_y)
    roughness = float(np.nanmedian(np.abs(z - np.median(z))))
    return {
        "cell_count": count,
        "filled_coverage": coverage,
        "raw_coverage": raw_coverage,
        "relief_m": relief,
        "roughness_m": roughness,
        "median_local_slope": float(np.nanmedian(slope_magnitude[filled])),
        "p95_local_slope": float(np.nanquantile(slope_magnitude[filled], 0.95)),
        "acceptance_rmse_m": limit,
        "plane_rmse_m": plane_rmse,
        "plane_slope_x": float(plane_beta[0]),
        "plane_slope_y": float(plane_beta[1]),
        "plane_intercept_m": float(plane_beta[2]),
        "cone_rmse_m": cone_rmse,
        "cone_slope": float(cone_beta[0]),
        "cone_intercept_m": float(cone_beta[1]),
        "cone_sector_slope_median": slope_median,
        "cone_sector_slope_mad": slope_mad,
        "cone_same_sign_fraction": same_sign_fraction,
        "sector_constant_accept_count": int(sector_constant_accept),
        "sector_constant_rmse_median": float(np.median(sector_constant_rmse)),
        **step,
        "support_ok": support_ok,
        "legal_cone": cone_candidate,
        "legal_plane": plane_candidate,
        "legal_step": step_candidate,
        "legal_sector": support_ok
        and sector_constant_accept == config.sector_count,
        "any_analytic_candidate": bool(
            cone_candidate
            or plane_candidate
            or step_candidate
            or sector_constant_accept == config.sector_count
        ),
        "geometry_status": "ok" if support_ok else "low_support",
        "geometry_config_json": str(asdict(config)),
    }
