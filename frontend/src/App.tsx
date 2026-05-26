import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ExportPayload,
  Source,
  bestInSlice,
  defaultSource,
  errorAt,
  simulateSensors,
  type Sensor,
  type SolverState,
} from "./math";

type CanvasProps = {
  sensors: Sensor[];
  source: Source;
  estimate?: Source;
  z: number;
  a0: number;
  best: { x: number; y: number; z: number; error: number };
};

function HeatmapCanvas({ sensors, source, estimate, z, a0, best }: CanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    requestAnimationFrame(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(rect.width * ratio);
      canvas.height = Math.floor(rect.height * ratio);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(ratio, ratio);
      draw(ctx, rect.width, rect.height, sensors, source, estimate, z, a0, best);
    });
  }, [sensors, source, estimate, z, a0, best]);

  return <canvas ref={canvasRef} className="heatmap" />;
}

function draw(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  sensors: Sensor[],
  source: Source,
  estimate: Source | undefined,
  z: number,
  a0: number,
  best: { x: number; y: number; z: number; error: number },
) {
  const pad = 42;
  const bounds = { min: -5.8, max: 5.8 };
  const sx = (x: number) => pad + ((x - bounds.min) / (bounds.max - bounds.min)) * (width - 2 * pad);
  const sy = (y: number) => height - pad - ((y - bounds.min) / (bounds.max - bounds.min)) * (height - 2 * pad);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);

  const cells = 58;
  const values: Array<{ x: number; y: number; error: number }> = [];
  for (let yi = 0; yi < cells; yi += 1) {
    const y = bounds.min + ((bounds.max - bounds.min) * yi) / (cells - 1);
    for (let xi = 0; xi < cells; xi += 1) {
      const x = bounds.min + ((bounds.max - bounds.min) * xi) / (cells - 1);
      values.push({ x, y, error: errorAt(sensors, { x0: x, y0: y, z0: z, a0 }) });
    }
  }
  const minError = Math.min(...values.map((item) => item.error));
  const maxError = Math.max(...values.map((item) => item.error));
  const cellW = (width - 2 * pad) / cells + 1;
  const cellH = (height - 2 * pad) / cells + 1;

  values.forEach((item) => {
    const t = (item.error - minError) / (maxError - minError + 1e-12);
    ctx.fillStyle = heatColor(t);
    ctx.fillRect(sx(item.x) - cellW / 2, sy(item.y) - cellH / 2, cellW, cellH);
  });

  ctx.strokeStyle = "#475569";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, width - 2 * pad, height - 2 * pad);

  ctx.fillStyle = "#0f172a";
  ctx.font = "12px Inter, system-ui, sans-serif";
  sensors.forEach((sensor) => {
    const x = sx(sensor.x);
    const y = sy(sensor.y);
    ctx.beginPath();
    ctx.moveTo(x, y - 7);
    ctx.lineTo(x - 7, y + 6);
    ctx.lineTo(x + 7, y + 6);
    ctx.closePath();
    ctx.fill();
    ctx.fillText(String(sensor.sensor_id), x + 9, y + 4);
  });

  drawCross(ctx, sx(source.x0), sy(source.y0), "#dc2626");
  drawCircle(ctx, sx(best.x), sy(best.y), "#16a34a");
  if (estimate) drawCircle(ctx, sx(estimate.x0), sy(estimate.y0), "#2563eb");
}

function heatColor(t: number) {
  const r = Math.round(32 + 212 * t);
  const g = Math.round(82 + 90 * (1 - Math.abs(t - 0.42)));
  const b = Math.round(190 - 136 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function drawCross(ctx: CanvasRenderingContext2D, x: number, y: number, color: string) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x - 8, y - 8);
  ctx.lineTo(x + 8, y + 8);
  ctx.moveTo(x - 8, y + 8);
  ctx.lineTo(x + 8, y - 8);
  ctx.stroke();
}

function drawCircle(ctx: CanvasRenderingContext2D, x: number, y: number, color: string) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.stroke();
}

export default function App() {
  const [source, setSource] = useState<Source>(defaultSource);
  const [sensors, setSensors] = useState<Sensor[]>(() => simulateSensors(defaultSource));
  const [solver, setSolver] = useState<SolverState[]>([]);
  const [z, setZ] = useState(defaultSource.z0);
  const best = useMemo(() => bestInSlice(sensors, z, source.a0), [sensors, z, source.a0]);
  const estimate = solver.length > 0 ? solver[solver.length - 1].fuente : undefined;

  const updateSource = (key: keyof Source, value: number) => {
    if (key === "z0") {
      const clamped = Math.min(10, Math.max(0.2, parseFloat(value.toFixed(2))));
      setSource((prev) => ({ ...prev, z0: clamped }));
      setZ(clamped);
    } else {
      setSource((prev) => ({ ...prev, [key]: value }));
    }
  };

  const loadJson = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const payload = JSON.parse(await file.text()) as ExportPayload;
    setSource(payload.fuente_real);
    setSensors(payload.sensores);
    setSolver(payload.solver ?? []);
    setZ(payload.analisis?.global_minimum?.z ?? payload.fuente_real.z0);
  };

  return (
    <main className="shell">
      <aside className="sidebar">
        <h1>Localizacion sismica</h1>
        <label>
          JSON Python
          <input type="file" accept="application/json" onChange={loadJson} />
        </label>
        {(["x0", "y0", "z0", "a0"] as Array<keyof Source>).map((key) => (
          <label key={key}>
            {key}
            <input
              type="number"
              step={key === "a0" ? 10 : 0.1}
              value={source[key]}
              min={key === "z0" ? 0.2 : undefined}
              max={key === "z0" ? 10 : undefined}
              onChange={(event) => updateSource(key, Number(event.target.value))}
            />
          </label>
        ))}
        <label>
          Corte z — {z.toFixed(2)}
          <div className="z-controls">
            <button
              className="z-btn"
              onClick={() => setZ((prev) => Math.max(0.2, parseFloat((prev - 0.1).toFixed(2))))}
            >
              −
            </button>
            <input type="range" min="0.2" max="10" step="0.02" value={z} onChange={(event) => setZ(Number(event.target.value))} />
            <button
              className="z-btn"
              onClick={() => setZ((prev) => Math.min(10, parseFloat((prev + 0.1).toFixed(2))))}
            >
              +
            </button>
          </div>
        </label>
        <button onClick={() => setSensors(simulateSensors(source))}>Simular sensores</button>
      </aside>

      <section className="workspace">
        <div className="metrics">
          <div>
            <span>Corte Z actual</span>
            <strong>{z.toFixed(2)} <small>/ 10.00</small></strong>
            <progress className="z-progress" value={z} max={10} />
          </div>
          <div>
            <span>Mejor punto del corte</span>
            <strong>
              ({best.x.toFixed(2)}, {best.y.toFixed(2)}, {best.z.toFixed(2)})
            </strong>
          </div>
          <div>
            <span>Error minimo</span>
            <strong>{best.error.toExponential(3)}</strong>
          </div>
          <div>
            <span>Estimacion iterativa</span>
            <strong>
              {estimate ? `${estimate.x0.toFixed(2)}, ${estimate.y0.toFixed(2)}, ${estimate.z0.toFixed(2)}` : "sin solver"}
            </strong>
          </div>
        </div>
        <HeatmapCanvas sensors={sensors} source={source} estimate={estimate} z={z} a0={source.a0} best={best} />
        <div className="legend">
          <span><i className="real" /> fuente real</span>
          <span><i className="grid" /> minimo del corte</span>
          <span><i className="estimate" /> estimacion solver</span>
          <span><i className="sensor" /> sensores</span>
        </div>
      </section>
    </main>
  );
}
