from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Sensor:
    sensor_id: int
    x: float
    y: float
    z: float
    amplitud_obs: float = 0.0

    def posicion(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class FuenteSismica:
    x0: float
    y0: float
    z0: float
    a0: float

    def vector(self) -> list[float]:
        return [self.x0, self.y0, self.z0, self.a0]

    @classmethod
    def from_vector(cls, values: Iterable[float]) -> "FuenteSismica":
        x0, y0, z0, a0 = values
        return cls(float(x0), float(y0), float(z0), float(a0))


@dataclass
class RedSensores:
    sensores: list[Sensor]

    @property
    def m(self) -> int:
        return len(self.sensores)

    def amplitudes_observadas(self) -> list[float]:
        return [sensor.amplitud_obs for sensor in self.sensores]

    def posiciones(self) -> list[tuple[float, float, float]]:
        return [sensor.posicion() for sensor in self.sensores]

    def asignar_amplitudes(self, amplitudes: Iterable[float]) -> None:
        for sensor, amplitud in zip(self.sensores, amplitudes):
            sensor.amplitud_obs = float(amplitud)
