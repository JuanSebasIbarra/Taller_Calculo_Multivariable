from __future__ import annotations

import math
import random

from .domain import FuenteSismica, RedSensores, Sensor
from .model import ModeloAtenuacion


def default_sensor_network() -> RedSensores:
    coordinates = [
        (-4.5, -3.0, 0.20),
        (-2.5, -4.0, 0.10),
        (0.0, -4.5, 0.15),
        (2.5, -3.5, 0.05),
        (4.5, -2.2, 0.12),
        (-5.0, 0.0, 0.18),
        (5.0, 0.2, 0.08),
        (-3.5, 3.2, 0.22),
        (-0.5, 4.4, 0.10),
        (3.2, 3.5, 0.16),
        (0.0, 0.0, 0.05),
        (1.8, -0.8, 0.09),
    ]
    return RedSensores([Sensor(i + 1, x, y, z) for i, (x, y, z) in enumerate(coordinates)])


class SimuladorDatos:
    def __init__(self, alpha: float = 0.05, semilla: int = 7) -> None:
        self.alpha = alpha
        self.semilla = semilla
        self.modelo = ModeloAtenuacion()

    def simular_amplitudes(self, red: RedSensores, fuente: FuenteSismica) -> list[float]:
        rng = random.Random(self.semilla)
        amplitudes: list[float] = []
        for sensor in red.sensores:
            limpia = self.modelo.amplitud_sensor(sensor, fuente)
            sigma = abs(self.alpha * limpia)
            amplitudes.append(limpia + rng.gauss(0.0, sigma))
        red.asignar_amplitudes(amplitudes)
        return amplitudes

    def fuente_por_defecto(self) -> FuenteSismica:
        return FuenteSismica(x0=1.2, y0=-1.1, z0=2.2, a0=950.0)

    def generar_red_circular(self, cantidad: int = 14, radio: float = 5.0) -> RedSensores:
        sensores: list[Sensor] = []
        for i in range(cantidad):
            angle = 2 * math.pi * i / cantidad
            z = 0.08 + 0.12 * (1 + math.sin(2 * angle)) / 2
            sensores.append(Sensor(i + 1, radio * math.cos(angle), radio * math.sin(angle), z))
        return RedSensores(sensores)
