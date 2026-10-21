"""Numerical configuration and radial helpers for bounded terrain evaluation."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from .geometry import GeometryConfig, huber_lstsq

@dataclass(frozen=True)
class CandidateConfig:
    density_kg_m3: float = 2200.0
    radius_m: float = 20.0
    sector_count: int = 8
    cone_preference_ratio: float = 0.90
    minimum_cone_slope: float = 0.005
    radial_split_improvement_ratio: float = 0.70
    radial_split_min_radius_m: float = 4.0
    radial_split_edge_margin_m: float = 4.0
    intercept_limit_floor_m: float = 0.15
    intercept_limit_relief_fraction: float = 0.10
    checkpoint_batch_size: int = 25
    prism_consistency_tolerance_mgal: float = 1e-10
    geometry: GeometryConfig = GeometryConfig()

    def validate(self) -> None:
        if self.density_kg_m3 <= 0 or self.radius_m <= 0:
            raise ValueError("density and radius must be positive")
        if self.sector_count <= 0:
            raise ValueError("sector_count must be positive")
        if self.checkpoint_batch_size <= 0:
            raise ValueError("checkpoint_batch_size must be positive")
        if self.intercept_limit_floor_m < 0 or self.intercept_limit_relief_fraction < 0:
            raise ValueError("intercept limits must be non-negative")
        if not math.isclose(self.geometry.radius_m, self.radius_m, abs_tol=1e-12):
            raise ValueError("Geometry and candidate radii must match")
        if self.geometry.sector_count != self.sector_count:
            raise ValueError("Geometry and candidate sector counts must match")


def _fit_radial_or_constant(
    radius: np.ndarray,
    height: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    beta, radial_rmse = huber_lstsq(
        np.column_stack([radius, np.ones_like(radius)]),
        height,
    )
    constant = float(np.median(height))
    constant_rmse = float(np.sqrt(np.mean((height - constant) ** 2)))
    return radial_rmse, constant_rmse, beta


def _radial_parts(
    radius: np.ndarray,
    height: np.ndarray,
    config: CandidateConfig,
) -> list[tuple[float, float]]:
    if len(height) < 40:
        return [(0.0, config.radius_m + 1e-9)]
    order = np.argsort(radius)
    sorted_radius = np.asarray(radius, dtype=float)[order]
    sorted_height = np.asarray(height, dtype=float)[order]
    cumulative_radius = np.cumsum(sorted_radius)
    cumulative_height = np.cumsum(sorted_height)
    cumulative_radius2 = np.cumsum(sorted_radius * sorted_radius)
    cumulative_height2 = np.cumsum(sorted_height * sorted_height)
    cumulative_cross = np.cumsum(sorted_radius * sorted_height)

    def segment_rmse(start_index: int, stop_index: int) -> float:
        count = stop_index - start_index
        if count < 3:
            return math.inf

        def interval(cumulative: np.ndarray) -> float:
            return float(
                cumulative[stop_index - 1]
                - (cumulative[start_index - 1] if start_index else 0.0)
            )

        sum_x = interval(cumulative_radius)
        sum_y = interval(cumulative_height)
        sum_x2 = interval(cumulative_radius2)
        sum_y2 = interval(cumulative_height2)
        sum_xy = interval(cumulative_cross)
        constant_sse = max(0.0, sum_y2 - sum_y * sum_y / count)
        denominator = count * sum_x2 - sum_x * sum_x
        if abs(denominator) <= 1e-12:
            return math.sqrt(constant_sse / count)
        slope = (count * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / count
        linear_sse = (
            sum_y2
            + slope * slope * sum_x2
            + count * intercept * intercept
            + 2.0 * slope * intercept * sum_x
            - 2.0 * slope * sum_xy
            - 2.0 * intercept * sum_y
        )
        return math.sqrt(max(0.0, min(constant_sse, linear_sse)) / count)

    baseline = segment_rmse(0, len(sorted_height))
    if not np.isfinite(baseline) or baseline <= 1e-8:
        return [(0.0, config.radius_m + 1e-9)]
    best_rmse = math.inf
    best_radius = None
    start = config.radial_split_min_radius_m
    stop = config.radius_m - config.radial_split_edge_margin_m
    for split in np.arange(start, stop + 1e-9, 1.0):
        split_index = int(np.searchsorted(sorted_radius, split, side="left"))
        if split_index < 15 or len(sorted_radius) - split_index < 15:
            continue
        inner_rmse = segment_rmse(0, split_index)
        outer_rmse = segment_rmse(split_index, len(sorted_radius))
        combined = math.sqrt(
            (
                inner_rmse**2 * split_index
                + outer_rmse**2 * (len(sorted_radius) - split_index)
            )
            / len(sorted_radius)
        )
        if combined < best_rmse:
            best_rmse = combined
            best_radius = float(split)
    if (
        best_radius is not None
        and best_rmse <= config.radial_split_improvement_ratio * baseline
    ):
        return [
            (0.0, best_radius),
            (best_radius, config.radius_m + 1e-9),
        ]
    return [(0.0, config.radius_m + 1e-9)]
