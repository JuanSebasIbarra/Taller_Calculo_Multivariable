# Clases principales

## Dominio

- `Sensor`: almacena `sensor_id`, posicion `(x,y,z)` y `amplitud_obs`.
- `RedSensores`: agrupa sensores y entrega posiciones o amplitudes observadas.
- `FuenteSismica`: representa el vector de parametros `m = [x0, y0, z0, A0]`.

## Modelo y simulacion

- `ModeloAtenuacion`: calcula distancia, amplitud predicha, residual, error y Jacobiano.
- `SimuladorDatos`: genera amplitudes sinteticas con ruido gaussiano moderado.

## Analisis e inversion

- `build_analysis`: explora cortes `z=k`, genera grillas `E(x,y,z)` y calcula `E_min(z)`.
- `SolverInverso`: aplica el metodo iterativo de minimos cuadrados.
- `IterationState`: guarda cada iteracion con fuente estimada, error y norma del paso.

## Interfaces

- `SeismicTkApp`: orquesta simulacion, analisis, solver, dibujo en canvas y exportacion JSON.
- `frontend/src/App.tsx`: visualiza cortes, sensores, fuente real, minimo por plano y resultado del solver.
