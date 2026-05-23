# Pseudocodigo del flujo sismico

```text
Inicio

Fase 1: simulacion de datos sinteticos
  definir red de sensores con posiciones (xi, yi, zi)
  definir fuente real (x0_real, y0_real, z0_real, A0_real)
  para cada sensor i:
    Ri = sqrt((xi-x0)^2 + (yi-y0)^2 + (zi-z0)^2)
    A_limpia_i = A0 * exp(-Ri) / Ri
    sigma_i = alpha * abs(A_limpia_i)
    Azi = A_limpia_i + ruido_gaussiano(mu=0, sigma=sigma_i)
  guardar Az observado en cada sensor

Fase 2: exploracion de la funcion de error
  definir rangos X, Y, Z
  para cada plano z = k:
    para cada punto (x, y) de la grilla:
      calcular amplitudes predichas A'z
      E(x,y,k) = suma((Az_i - A'z_i)^2)
    identificar (x*, y*) que minimiza el plano
    guardar E_min(k) = min E(x,y,k)
  seleccionar z* con menor E_min(z)

Fase 3: solucion iterativa por minimos cuadrados
  inicializar m0 = [x0, y0, z0, A0]
  repetir hasta convergencia:
    calcular A'z(mk)
    calcular residual DeltaAz = Az - A'z(mk)
    construir Jacobiano G con derivadas respecto a x0, y0, z0, A0
    resolver DeltaM = (G^T G)^-1 G^T DeltaAz
    actualizar mk+1 = mk + DeltaM
    evaluar Err = suma((Az_i - A'z_i)^2)
  reportar m* = [x0*, y0*, z0*, A0*]

Fin
```
