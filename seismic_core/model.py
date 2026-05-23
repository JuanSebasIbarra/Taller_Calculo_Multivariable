from __future__ import annotations

import math

from .domain import FuenteSismica, RedSensores, Sensor


class ModeloAtenuacion:
    """Modelo A = A0 * exp(-R) / R usado en el enunciado del proyecto."""

    def __init__(self, distancia_minima: float = 1e-6) -> None:
        self.distancia_minima = distancia_minima

    def distancia(self, sensor: Sensor, fuente: FuenteSismica) -> float:
        dx = sensor.x - fuente.x0
        dy = sensor.y - fuente.y0
        dz = sensor.z - fuente.z0
        return max(math.sqrt(dx * dx + dy * dy + dz * dz), self.distancia_minima)

    def amplitud_sensor(self, sensor: Sensor, fuente: FuenteSismica) -> float:
        r = self.distancia(sensor, fuente)
        return fuente.a0 * math.exp(-r) / r

    def predecir(self, red: RedSensores, fuente: FuenteSismica) -> list[float]:
        return [self.amplitud_sensor(sensor, fuente) for sensor in red.sensores]

    def error(self, red: RedSensores, fuente: FuenteSismica) -> float:
        predichas = self.predecir(red, fuente)
        observadas = red.amplitudes_observadas()
        return sum((obs - pred) ** 2 for obs, pred in zip(observadas, predichas))

    def residual(self, red: RedSensores, fuente: FuenteSismica) -> list[float]:
        predichas = self.predecir(red, fuente)
        return [obs - pred for obs, pred in zip(red.amplitudes_observadas(), predichas)]

    def jacobiano(self, red: RedSensores, fuente: FuenteSismica) -> list[list[float]]:
        rows: list[list[float]] = []
        for sensor in red.sensores:
            r = self.distancia(sensor, fuente)
            exp_term = math.exp(-r)
            df_dr = -exp_term * (1.0 / r + 1.0 / (r * r))
            d_r_dx0 = (fuente.x0 - sensor.x) / r
            d_r_dy0 = (fuente.y0 - sensor.y) / r
            d_r_dz0 = (fuente.z0 - sensor.z) / r
            rows.append(
                [
                    fuente.a0 * df_dr * d_r_dx0,
                    fuente.a0 * df_dr * d_r_dy0,
                    fuente.a0 * df_dr * d_r_dz0,
                    exp_term / r,
                ]
            )
        return rows
