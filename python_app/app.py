from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from seismic_core import (
    FuenteSismica,
    ModeloAtenuacion,
    SimuladorDatos,
    SolverInverso,
    build_analysis,
    default_sensor_network,
)


ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = ROOT / "data" / "exports" / "resultado_sismico.json"


class SeismicTkApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Localizacion de fuente sismica")
        self.geometry("1120x720")
        self.minsize(980, 620)

        self.modelo = ModeloAtenuacion()
        self.simulador = SimuladorDatos()
        self.red = default_sensor_network()
        self.fuente_real = self.simulador.fuente_por_defecto()
        self.estados = []
        self.analisis: dict | None = None

        self.vars = {
            "x0": tk.DoubleVar(value=self.fuente_real.x0),
            "y0": tk.DoubleVar(value=self.fuente_real.y0),
            "z0": tk.DoubleVar(value=self.fuente_real.z0),
            "a0": tk.DoubleVar(value=self.fuente_real.a0),
            "ix": tk.DoubleVar(value=0.0),
            "iy": tk.DoubleVar(value=0.0),
            "iz": tk.DoubleVar(value=1.0),
            "ia": tk.DoubleVar(value=700.0),
            "z_slice": tk.DoubleVar(value=2.2),
        }

        self._build_ui()
        self._simulate()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="ns")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Fuente real").grid(row=0, column=0, columnspan=2, sticky="w")
        self._entry(controls, "x0", "x real", 1)
        self._entry(controls, "y0", "y real", 2)
        self._entry(controls, "z0", "z real", 3)
        self._entry(controls, "a0", "A0 real", 4)

        ttk.Separator(controls).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Label(controls, text="Estimacion inicial").grid(row=6, column=0, columnspan=2, sticky="w")
        self._entry(controls, "ix", "x inicial", 7)
        self._entry(controls, "iy", "y inicial", 8)
        self._entry(controls, "iz", "z inicial", 9)
        self._entry(controls, "ia", "A0 inicial", 10)

        ttk.Separator(controls).grid(row=11, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Label(controls, text="Corte z").grid(row=12, column=0, sticky="w")
        z_scale = ttk.Scale(controls, from_=0.2, to=5.0, variable=self.vars["z_slice"], command=self._draw)
        z_scale.grid(row=12, column=1, sticky="ew")

        ttk.Button(controls, text="Simular datos", command=self._simulate).grid(
            row=13, column=0, columnspan=2, pady=(14, 4), sticky="ew"
        )
        ttk.Button(controls, text="Analizar E(x,y,z)", command=self._analyze).grid(
            row=14, column=0, columnspan=2, pady=4, sticky="ew"
        )
        ttk.Button(controls, text="Resolver inverso", command=self._solve).grid(
            row=15, column=0, columnspan=2, pady=4, sticky="ew"
        )
        ttk.Button(controls, text="Exportar JSON", command=self._export).grid(
            row=16, column=0, columnspan=2, pady=4, sticky="ew"
        )

        self.summary = tk.Text(controls, width=35, height=18, wrap="word")
        self.summary.grid(row=17, column=0, columnspan=2, pady=(12, 0), sticky="nsew")

        workspace = ttk.Frame(self, padding=(0, 12, 12, 12))
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(workspace, background="#f6f7f8", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

    def _entry(self, parent: ttk.Frame, key: str, label: str, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(row=row, column=1, sticky="ew", pady=2)

    def _read_real_source(self) -> FuenteSismica:
        return FuenteSismica(
            self.vars["x0"].get(),
            self.vars["y0"].get(),
            self.vars["z0"].get(),
            self.vars["a0"].get(),
        )

    def _read_initial_source(self) -> FuenteSismica:
        return FuenteSismica(
            self.vars["ix"].get(),
            self.vars["iy"].get(),
            self.vars["iz"].get(),
            self.vars["ia"].get(),
        )

    def _simulate(self) -> None:
        self.fuente_real = self._read_real_source()
        self.simulador.simular_amplitudes(self.red, self.fuente_real)
        self.analisis = None
        self.estados = []
        self._write_summary("Datos simulados con ruido gaussiano alpha=0.05.")
        self._draw()

    def _analyze(self) -> None:
        self.analisis = build_analysis(
            self.red,
            self.modelo,
            a0=self.vars["a0"].get(),
            xy_count=55,
            z_count=80,
        )
        g = self.analisis["global_minimum"]
        self._write_summary(
            "Analisis de funcion de error completado.\n"
            f"Minimo global en grilla: x={g['x']:.3f}, y={g['y']:.3f}, z={g['z']:.3f}\n"
            f"Error minimo: {g['error']:.6g}"
        )
        self._draw()

    def _solve(self) -> None:
        solver = SolverInverso()
        self.estados = solver.resolver(self.red, self._read_initial_source())
        final = self.estados[-1]
        self._write_summary(
            "Solver iterativo finalizado.\n"
            f"Iteraciones: {final.iteracion}\n"
            f"Estimado: x={final.fuente.x0:.4f}, y={final.fuente.y0:.4f}, "
            f"z={final.fuente.z0:.4f}, A0={final.fuente.a0:.2f}\n"
            f"Error final: {final.error:.6g}\n"
            f"Delta final: {final.delta_norm:.6g}"
        )
        self._draw()

    def _export(self) -> None:
        EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload()
        EXPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._write_summary(f"Resultados exportados en:\n{EXPORT_PATH}")

    def _payload(self) -> dict:
        return {
            "fuente_real": self.fuente_real.__dict__,
            "sensores": [sensor.__dict__ for sensor in self.red.sensores],
            "solver": [
                {
                    "iteracion": state.iteracion,
                    "fuente": state.fuente.__dict__,
                    "error": state.error,
                    "delta_norm": state.delta_norm,
                }
                for state in self.estados
            ],
            "analisis": self.analisis,
        }

    def _write_summary(self, text: str) -> None:
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", text)

    def _draw(self, *_args) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 800)
        height = max(self.canvas.winfo_height(), 520)
        pad = 54
        plot_w = width - 2 * pad
        plot_h = height - 2 * pad
        x_min, x_max = -5.8, 5.8
        y_min, y_max = -5.8, 5.8

        def sx(x: float) -> float:
            return pad + (x - x_min) / (x_max - x_min) * plot_w

        def sy(y: float) -> float:
            return height - pad - (y - y_min) / (y_max - y_min) * plot_h

        self._draw_heatmap(sx, sy, x_min, x_max, y_min, y_max)
        self.canvas.create_rectangle(pad, pad, width - pad, height - pad, outline="#6b7280")
        self.canvas.create_text(pad, 22, anchor="w", text="Mapa de error E(x,y,z) y red de sensores")
        self.canvas.create_text(
            width - pad,
            22,
            anchor="e",
            text=f"z = {self.vars['z_slice'].get():.2f}",
        )

        for sensor in self.red.sensores:
            x, y = sx(sensor.x), sy(sensor.y)
            self.canvas.create_polygon(x, y - 7, x - 7, y + 6, x + 7, y + 6, fill="#111827", outline="")
            self.canvas.create_text(x + 10, y, anchor="w", text=str(sensor.sensor_id), fill="#374151")

        rx, ry = sx(self.fuente_real.x0), sy(self.fuente_real.y0)
        self.canvas.create_line(rx - 8, ry - 8, rx + 8, ry + 8, fill="#dc2626", width=3)
        self.canvas.create_line(rx - 8, ry + 8, rx + 8, ry - 8, fill="#dc2626", width=3)

        if self.estados:
            est = self.estados[-1].fuente
            ex, ey = sx(est.x0), sy(est.y0)
            self.canvas.create_oval(ex - 8, ey - 8, ex + 8, ey + 8, outline="#2563eb", width=3)

    def _draw_heatmap(self, sx, sy, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        count = 42
        z = self.vars["z_slice"].get()
        a0 = self.vars["a0"].get()
        values: list[tuple[float, float, float]] = []
        for yi in range(count):
            y = y_min + (y_max - y_min) * yi / (count - 1)
            for xi in range(count):
                x = x_min + (x_max - x_min) * xi / (count - 1)
                err = self.modelo.error(self.red, FuenteSismica(x, y, z, a0))
                values.append((x, y, err))
        min_e = min(v[2] for v in values)
        max_e = max(v[2] for v in values)
        dx = abs(sx(x_min + (x_max - x_min) / count) - sx(x_min)) + 1
        dy = abs(sy(y_min + (y_max - y_min) / count) - sy(y_min)) + 1
        for x, y, err in values:
            t = (err - min_e) / (max_e - min_e + 1e-12)
            color = self._heat_color(t)
            cx, cy = sx(x), sy(y)
            self.canvas.create_rectangle(cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2, fill=color, outline=color)

    @staticmethod
    def _heat_color(t: float) -> str:
        t = max(0.0, min(1.0, t))
        r = int(32 + 210 * t)
        g = int(70 + 100 * (1 - abs(t - 0.45)))
        b = int(180 - 135 * t)
        return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    SeismicTkApp().mainloop()
