from __future__ import annotations

from dataclasses import dataclass

from .domain import FuenteSismica, RedSensores
from .model import ModeloAtenuacion


@dataclass
class SliceMinimum:
    z: float
    x: float
    y: float
    error: float


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def grid_error_xy(
    red: RedSensores,
    modelo: ModeloAtenuacion,
    z: float,
    a0: float,
    x_values: list[float],
    y_values: list[float],
) -> list[list[float]]:
    return [
        [modelo.error(red, FuenteSismica(x, y, z, a0)) for x in x_values]
        for y in y_values
    ]


def minimum_in_grid(
    grid: list[list[float]], x_values: list[float], y_values: list[float], z: float
) -> SliceMinimum:
    best = SliceMinimum(z=z, x=x_values[0], y=y_values[0], error=grid[0][0])
    for yi, row in enumerate(grid):
        for xi, value in enumerate(row):
            if value < best.error:
                best = SliceMinimum(z=z, x=x_values[xi], y=y_values[yi], error=value)
    return best


def build_analysis(
    red: RedSensores,
    modelo: ModeloAtenuacion,
    a0: float,
    x_range: tuple[float, float] = (-5.5, 5.5),
    y_range: tuple[float, float] = (-5.5, 5.5),
    z_range: tuple[float, float] = (0.2, 5.0),
    xy_count: int = 45,
    z_count: int = 60,
) -> dict:
    x_values = linspace(x_range[0], x_range[1], xy_count)
    y_values = linspace(y_range[0], y_range[1], xy_count)
    z_values = linspace(z_range[0], z_range[1], z_count)
    minima: list[SliceMinimum] = []
    representative: dict[str, list[list[float]]] = {}

    for index, z in enumerate(z_values):
        grid = grid_error_xy(red, modelo, z, a0, x_values, y_values)
        minima.append(minimum_in_grid(grid, x_values, y_values, z))
        if index in {0, len(z_values) // 2, len(z_values) - 1}:
            representative[f"{z:.3f}"] = grid

    global_min = min(minima, key=lambda item: item.error)
    return {
        "x_values": x_values,
        "y_values": y_values,
        "z_values": z_values,
        "minima": [minimum.__dict__ for minimum in minima],
        "global_minimum": global_min.__dict__,
        "representative_grids": representative,
    }
