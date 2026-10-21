"""Analytical terrain models and the rectangular-prism numerical kernel."""

from __future__ import annotations

import math

import numpy as np


G = 6.67430e-11
SI_TO_MGAL = 1.0e5


def cone_correction_mgal(
    radius_m: float,
    inclination_rad: float,
    density_kg_m3: float,
) -> float:
    """Full-axisymmetric cone terrain-correction magnitude."""

    if radius_m <= 0 or density_kg_m3 <= 0:
        return 0.0
    value_si = (
        2.0
        * math.pi
        * G
        * density_kg_m3
        * radius_m
        * (1.0 - math.cos(abs(inclination_rad)))
    )
    return abs(value_si) * SI_TO_MGAL


def conical_sector_correction_mgal(
    radius_m: float,
    inclination_rad: float,
    density_kg_m3: float,
    sector_angle_rad: float,
) -> float:
    """Azimuth-limited cone correction by angular proportionality."""

    fraction = float(np.clip(abs(sector_angle_rad) / (2.0 * math.pi), 0.0, 1.0))
    return cone_correction_mgal(radius_m, inclination_rad, density_kg_m3) * fraction


def _plane_series(inclination_rad: float, terms: int = 64) -> float:
    sin2 = math.sin(abs(inclination_rad)) ** 2
    total = 0.0
    for index in range(terms):
        coefficient = math.comb(2 * index, index) / (4.0**index)
        total += coefficient * coefficient * (sin2**index)
    return total


def inclined_half_plane_correction_mgal(
    radius_m: float,
    inclination_rad: float,
    density_kg_m3: float,
) -> float:
    """Rising half of a circular inclined plane."""

    if radius_m <= 0 or density_kg_m3 <= 0:
        return 0.0
    inclination = abs(inclination_rad)
    bracket = 1.0 - math.cos(inclination) * _plane_series(inclination)
    return abs(math.pi * G * density_kg_m3 * radius_m * bracket) * SI_TO_MGAL


def centred_inclined_plane_correction_mgal(
    radius_m: float,
    inclination_rad: float,
    density_kg_m3: float,
) -> float:
    """Both halves of a centred inclined circular plane."""

    return 2.0 * inclined_half_plane_correction_mgal(
        radius_m,
        inclination_rad,
        density_kg_m3,
    )


def straight_step_correction_mgal(
    perpendicular_distance_m: float,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
) -> float:
    """Straight single-step terrain-correction magnitude."""

    distance = float(perpendicular_distance_m)
    radius = float(radius_m)
    height = abs(float(height_m))
    if height == 0 or distance <= 0 or radius <= distance or density_kg_m3 <= 0:
        return 0.0
    alpha = math.acos(float(np.clip(distance / radius, -1.0, 1.0)))
    sec_plus_tan = 1.0 / math.cos(alpha) + math.tan(alpha)
    log_argument = (
        2.0 * distance * math.sqrt(radius * radius + height * height) * math.tan(alpha)
        + 2.0 * radius * radius
        + height * height
        - distance * distance
    )
    asin_argument = (
        (height * height - distance * distance) * radius * radius / (distance * distance)
        - 2.0 * height * height
    ) / (
        (height * height + distance * distance) * radius * radius / (distance * distance)
    )
    bracket = (
        alpha * (radius - math.sqrt(radius * radius + height * height))
        - distance * math.log(max(sec_plus_tan, 1e-15))
        + 0.5 * distance * math.log(max(log_argument, 1e-15))
        + 0.5 * height * math.asin(float(np.clip(asin_argument, -1.0, 1.0)))
        - 0.5 * distance * math.log(height * height + distance * distance)
        + math.pi * height / 4.0
    )
    return abs(2.0 * G * density_kg_m3 * bracket) * SI_TO_MGAL


def sector_correction_mgal(
    inner_radius_m: float,
    outer_radius_m: float,
    height_difference_m: float,
    density_kg_m3: float,
    sector_angle_rad: float,
) -> float:
    """Constant-height annular-sector terrain-correction magnitude."""

    inner = max(0.0, float(inner_radius_m))
    outer = float(outer_radius_m)
    height = abs(float(height_difference_m))
    if outer <= inner or height == 0 or density_kg_m3 <= 0:
        return 0.0
    radial_term = (
        outer
        - inner
        + math.sqrt(inner * inner + height * height)
        - math.sqrt(outer * outer + height * height)
    )
    return (
        abs(G * density_kg_m3 * abs(sector_angle_rad) * radial_term)
        * SI_TO_MGAL
    )


def prism_vertical_si(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    density_kg_m3: np.ndarray | float,
) -> np.ndarray:
    """Nagy rectangular-prism vertical attraction in m/s²."""

    total = np.zeros(np.broadcast(x1, y1, z1).shape, dtype=float)
    for x_index, x in enumerate((x1, x2)):
        for y_index, y in enumerate((y1, y2)):
            for z_index, z in enumerate((z1, z2)):
                radius = np.sqrt(x * x + y * y + z * z)
                term = x * np.log(np.maximum(y + radius, 1e-15))
                term += y * np.log(np.maximum(x + radius, 1e-15))
                numerator = np.asarray(x * y, dtype=float)
                denominator = np.asarray(z * radius, dtype=float)
                numerator, denominator = np.broadcast_arrays(numerator, denominator)
                ratio = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator, dtype=float),
                    where=np.abs(denominator) > 1e-15,
                )
                singular = np.abs(denominator) <= 1e-15
                ratio = np.where(singular & (numerator > 0.0), np.inf, ratio)
                ratio = np.where(singular & (numerator < 0.0), -np.inf, ratio)
                term -= z * np.arctan(ratio)
                sign = -1.0 if (x_index + y_index + z_index) % 2 else 1.0
                total += sign * term
    return G * np.asarray(density_kg_m3) * total


def numerical_grid_correction_mgal(
    x: np.ndarray,
    y: np.ndarray,
    relative_height_m: np.ndarray,
    mask: np.ndarray,
    cell_size_m: float,
    density_kg_m3: float,
) -> float:
    """Sum cell-prism correction magnitudes over a square-cell terrain grid."""

    use = np.asarray(mask, dtype=bool) & np.isfinite(relative_height_m)
    if not np.any(use):
        return math.nan
    x_use = np.asarray(x, dtype=float)[use]
    y_use = np.asarray(y, dtype=float)[use]
    height = np.asarray(relative_height_m, dtype=float)[use]
    half = float(cell_size_m) / 2.0
    signed_density = np.where(height >= 0.0, density_kg_m3, -density_kg_m3)
    values = prism_vertical_si(
        x_use - half,
        x_use + half,
        y_use - half,
        y_use + half,
        np.minimum(height, 0.0),
        np.maximum(height, 0.0),
        signed_density,
    )
    return float(np.sum(np.abs(values)) * SI_TO_MGAL)


def numerical_grid_annulus_correction_mgal(
    x: np.ndarray,
    y: np.ndarray,
    relative_height_m: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_m: float,
    density_kg_m3: float,
    inner_radius_m: float,
    outer_radius_m: float,
    *,
    boundary_subdivisions: int = 8,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, float | int]]:
    """Integrate a circular annulus and subdivide only intersected boundary cells.

    Whole cells that are provably inside the annulus use the standard Nagy
    prism kernel.  Cells that can intersect either circular boundary are split
    into equal square sub-prisms; sub-prism centres determine the retained area.
    This avoids treating a 12.5 m cell as wholly present or absent at a 20 m
    boundary while retaining efficient whole-cell evaluation away from it.
    """

    if cell_size_m <= 0 or density_kg_m3 <= 0:
        raise ValueError("cell size and density must be positive")
    if inner_radius_m < 0 or outer_radius_m <= inner_radius_m:
        raise ValueError("annulus radii must satisfy 0 <= inner < outer")
    if boundary_subdivisions < 1:
        raise ValueError("boundary_subdivisions must be positive")
    x_values, y_values, heights, valid = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(relative_height_m, dtype=float),
        np.asarray(valid_mask, dtype=bool),
    )
    valid = valid & np.isfinite(heights)
    radius = np.hypot(x_values, y_values)
    half_diagonal = float(cell_size_m) / math.sqrt(2.0)
    certainly_inside_outer = radius + half_diagonal <= outer_radius_m
    certainly_outside_inner = (
        np.ones(radius.shape, dtype=bool)
        if inner_radius_m == 0
        else radius - half_diagonal > inner_radius_m
    )
    whole = valid & certainly_inside_outer & certainly_outside_inner
    possible = valid & (radius - half_diagonal <= outer_radius_m)
    if inner_radius_m > 0:
        possible &= radius + half_diagonal > inner_radius_m
    boundary = possible & ~whole

    total_mgal = 0.0
    if np.any(whole):
        total_mgal += numerical_grid_correction_mgal(
            x_values,
            y_values,
            heights,
            whole,
            cell_size_m,
            density_kg_m3,
        )

    boundary_indices = np.nonzero(boundary)
    retained_subcells = 0
    total_subcells = int(np.count_nonzero(boundary)) * boundary_subdivisions**2
    if total_subcells:
        offset_axis = (
            (np.arange(boundary_subdivisions, dtype=float) + 0.5)
            / boundary_subdivisions
            - 0.5
        ) * cell_size_m
        offset_x, offset_y = np.meshgrid(offset_axis, offset_axis)
        centre_x = x_values[boundary_indices][:, None, None]
        centre_y = y_values[boundary_indices][:, None, None]
        sub_x = centre_x + offset_x
        sub_y = centre_y + offset_y
        sub_radius = np.hypot(sub_x, sub_y)
        retain = (sub_radius > inner_radius_m) & (sub_radius <= outer_radius_m)
        retained_subcells = int(np.count_nonzero(retain))
        if retained_subcells:
            repeated_heights = np.broadcast_to(
                heights[boundary_indices][:, None, None], sub_x.shape
            )
            sub_size = cell_size_m / boundary_subdivisions
            total_mgal += numerical_grid_correction_mgal(
                sub_x,
                sub_y,
                repeated_heights,
                retain,
                sub_size,
                density_kg_m3,
            )

    diagnostics: dict[str, float | int] = {
        "whole_cell_count": int(np.count_nonzero(whole)),
        "boundary_cell_count": int(np.count_nonzero(boundary)),
        "boundary_subdivisions": int(boundary_subdivisions),
        "retained_boundary_subcell_count": retained_subcells,
        "candidate_boundary_subcell_count": total_subcells,
        "effective_cell_area_m2": float(
            np.count_nonzero(whole) * cell_size_m**2
            + retained_subcells * (cell_size_m / boundary_subdivisions) ** 2
        ),
    }
    if return_diagnostics:
        return float(total_mgal), diagnostics
    return float(total_mgal)


def centred_grid(
    radius_m: float,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a cell-centred square grid and circular cell-centre mask."""

    count = int(math.ceil(radius_m / cell_size_m))
    axis = (np.arange(-count, count) + 0.5) * cell_size_m
    x, y = np.meshgrid(axis, axis)
    return x, y, np.hypot(x, y) <= radius_m
