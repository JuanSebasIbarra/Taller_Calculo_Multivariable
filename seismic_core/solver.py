from __future__ import annotations

from dataclasses import dataclass

from .domain import FuenteSismica, RedSensores
from .linear_algebra import normal_equations, solve_linear_system
from .model import ModeloAtenuacion


@dataclass
class IterationState:
    iteracion: int
    fuente: FuenteSismica
    error: float
    delta_norm: float


class SolverInverso:
    def __init__(
        self,
        modelo: ModeloAtenuacion | None = None,
        tolerancia: float = 1e-8,
        max_iter: int = 40,
        damping: float = 1e-5,
        paso_maximo: float = 1.0,
    ) -> None:
        self.modelo = modelo or ModeloAtenuacion()
        self.tolerancia = tolerancia
        self.max_iter = max_iter
        self.damping = damping
        self.paso_maximo = paso_maximo

    def resolver(self, red: RedSensores, inicial: FuenteSismica) -> list[IterationState]:
        estados = [IterationState(0, inicial, self.modelo.error(red, inicial), 0.0)]
        actual = inicial

        for k in range(1, self.max_iter + 1):
            residual = self.modelo.residual(red, actual)
            jacobiano = self.modelo.jacobiano(red, actual)
            lhs, rhs = normal_equations(jacobiano, residual, self.damping)
            delta = solve_linear_system(lhs, rhs)
            delta = self._limitar_paso(delta)
            siguiente = FuenteSismica(
                actual.x0 + delta[0],
                actual.y0 + delta[1],
                max(0.05, actual.z0 + delta[2]),
                max(1.0, actual.a0 + delta[3]),
            )
            delta_norm = sum(value * value for value in delta) ** 0.5
            error = self.modelo.error(red, siguiente)
            estados.append(IterationState(k, siguiente, error, delta_norm))
            actual = siguiente
            if delta_norm < self.tolerancia or abs(estados[-2].error - error) < self.tolerancia:
                break

        return estados

    def _limitar_paso(self, delta: list[float]) -> list[float]:
        spatial_norm = sum(value * value for value in delta[:3]) ** 0.5
        if spatial_norm <= self.paso_maximo:
            return delta
        factor = self.paso_maximo / spatial_norm
        return [delta[0] * factor, delta[1] * factor, delta[2] * factor, delta[3] * factor]
