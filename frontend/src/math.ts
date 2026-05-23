export type Sensor = {
  sensor_id: number;
  x: number;
  y: number;
  z: number;
  amplitud_obs: number;
};

export type Source = {
  x0: number;
  y0: number;
  z0: number;
  a0: number;
};

export type SolverState = {
  iteracion: number;
  fuente: Source;
  error: number;
  delta_norm: number;
};

export type ExportPayload = {
  fuente_real: Source;
  sensores: Sensor[];
  solver: SolverState[];
  analisis?: {
    global_minimum?: { x: number; y: number; z: number; error: number };
    minima?: Array<{ x: number; y: number; z: number; error: number }>;
  };
};

export const defaultSensors: Sensor[] = [
  [-4.5, -3, 0.2],
  [-2.5, -4, 0.1],
  [0, -4.5, 0.15],
  [2.5, -3.5, 0.05],
  [4.5, -2.2, 0.12],
  [-5, 0, 0.18],
  [5, 0.2, 0.08],
  [-3.5, 3.2, 0.22],
  [-0.5, 4.4, 0.1],
  [3.2, 3.5, 0.16],
  [0, 0, 0.05],
  [1.8, -0.8, 0.09],
].map(([x, y, z], index) => ({ sensor_id: index + 1, x, y, z, amplitud_obs: 0 }));

export const defaultSource: Source = { x0: 1.2, y0: -1.1, z0: 2.2, a0: 950 };

export function distance(sensor: Sensor, source: Source): number {
  const dx = sensor.x - source.x0;
  const dy = sensor.y - source.y0;
  const dz = sensor.z - source.z0;
  return Math.max(Math.hypot(dx, dy, dz), 1e-6);
}

export function amplitude(sensor: Sensor, source: Source): number {
  const r = distance(sensor, source);
  return (source.a0 * Math.exp(-r)) / r;
}

export function simulateSensors(source: Source, alpha = 0.05): Sensor[] {
  return defaultSensors.map((sensor, index) => {
    const clean = amplitude(sensor, source);
    const noise = clean * alpha * seededNoise(index + 11);
    return { ...sensor, amplitud_obs: clean + noise };
  });
}

export function errorAt(sensors: Sensor[], source: Source): number {
  return sensors.reduce((total, sensor) => {
    const diff = sensor.amplitud_obs - amplitude(sensor, source);
    return total + diff * diff;
  }, 0);
}

export function bestInSlice(
  sensors: Sensor[],
  z: number,
  a0: number,
  count = 52,
): { x: number; y: number; z: number; error: number } {
  let best = { x: -5.5, y: -5.5, z, error: Number.POSITIVE_INFINITY };
  for (let yi = 0; yi < count; yi += 1) {
    const y = -5.5 + (11 * yi) / (count - 1);
    for (let xi = 0; xi < count; xi += 1) {
      const x = -5.5 + (11 * xi) / (count - 1);
      const error = errorAt(sensors, { x0: x, y0: y, z0: z, a0 });
      if (error < best.error) best = { x, y, z, error };
    }
  }
  return best;
}

function seededNoise(seed: number): number {
  const a = Math.sin(seed * 12.9898) * 43758.5453;
  const b = Math.sin((seed + 1) * 78.233) * 24634.6345;
  return ((a - Math.floor(a)) + (b - Math.floor(b)) - 1) * 1.6;
}
