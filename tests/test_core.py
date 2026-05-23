import unittest

from seismic_core import FuenteSismica, ModeloAtenuacion, SimuladorDatos, SolverInverso, default_sensor_network


class CoreTest(unittest.TestCase):
    def test_error_is_near_zero_without_noise(self):
        red = default_sensor_network()
        fuente = FuenteSismica(1.2, -1.1, 2.2, 950.0)
        modelo = ModeloAtenuacion()
        red.asignar_amplitudes(modelo.predecir(red, fuente))

        self.assertLess(modelo.error(red, fuente), 1e-20)

    def test_solver_reduces_error(self):
        red = default_sensor_network()
        fuente = FuenteSismica(1.2, -1.1, 2.2, 950.0)
        SimuladorDatos(alpha=0.02, semilla=3).simular_amplitudes(red, fuente)
        inicial = FuenteSismica(0.0, 0.0, 1.0, 700.0)
        estados = SolverInverso(max_iter=25).resolver(red, inicial)

        self.assertLess(estados[-1].error, estados[0].error)
        self.assertGreaterEqual(estados[-1].fuente.z0, 0.05)
        self.assertGreater(estados[-1].fuente.a0, 1.0)


if __name__ == "__main__":
    unittest.main()
