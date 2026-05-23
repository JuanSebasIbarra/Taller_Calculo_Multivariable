# Localizacion de una fuente sismica

Proyecto de calculo multivariable para estimar la posicion de una fuente sismica a partir de amplitudes medidas por sensores.

El modelo implementado sigue el PDF:

```text
A_i = A0 * exp(-R_i) / R_i + ruido
R_i = sqrt((xi - x0)^2 + (yi - y0)^2 + (zi - z0)^2)
Err = sum((A_observada_i - A_predicha_i)^2)
```

## Estructura

- `seismic_core/`: modelo matematico, simulacion, funcion de error, analisis por cortes y solver inverso.
- `python_app/`: interfaz de escritorio con Tkinter.
- `frontend/`: visualizador en TypeScript con React.
- `docs/`: pseudocodigo y clases usadas para conectar el diseno con la implementacion.
- `data/exports/`: salida JSON generada por la app Tkinter.

## Ejecutar Python con Tkinter

```bash
python3 main.py
```

Flujo sugerido:

1. Ajustar la fuente real o dejar los valores por defecto.
2. Presionar `Simular datos`.
3. Presionar `Analizar E(x,y,z)` para encontrar minimos por cortes en `z`.
4. Presionar `Resolver inverso` para estimar `x0`, `y0`, `z0` y `A0`.
5. Presionar `Exportar JSON` para usar los resultados en React.

## Ejecutar el visualizador React

```bash
cd frontend
npm install
npm run dev
```

El visualizador permite simular sensores en TypeScript o cargar el archivo exportado por Python:

```text
data/exports/resultado_sismico.json
```

## Resultado esperado

La solucion debe mostrar:

- Sensores como triangulos negros.
- Fuente real como una cruz roja.
- Minimo del corte `z = k` como un circulo verde.
- Estimacion iterativa como un circulo azul.
- Mapas de calor de la funcion de error `E(x,y,z)`.
