from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.geometry("1280x800")
        self.minsize(1050, 680)

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
        self.vars["z0"].trace_add("write", self._on_z0_changed)
        self._simulate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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
        ttk.Label(controls, text="Corte z (mapa de calor)").grid(row=12, column=0, columnspan=2, sticky="w")
        self.z_label = ttk.Label(controls, text=f"{self.vars['z_slice'].get():.2f} / 10.00")
        self.z_label.grid(row=12, column=1, sticky="e")
        z_scale = ttk.Scale(controls, from_=0.2, to=10.0, variable=self.vars["z_slice"], command=self._on_z_scale)
        z_scale.grid(row=13, column=0, columnspan=2, sticky="ew")

        ttk.Button(controls, text="Simular datos", command=self._simulate).grid(
            row=14, column=0, columnspan=2, pady=(14, 4), sticky="ew"
        )
        ttk.Button(controls, text="Analizar E(x,y,z)", command=self._analyze).grid(
            row=15, column=0, columnspan=2, pady=4, sticky="ew"
        )
        ttk.Button(controls, text="Resolver inverso", command=self._solve).grid(
            row=16, column=0, columnspan=2, pady=4, sticky="ew"
        )
        ttk.Button(controls, text="Exportar JSON", command=self._export).grid(
            row=17, column=0, columnspan=2, pady=4, sticky="ew"
        )

        self.summary = tk.Text(controls, width=35, height=16, wrap="word")
        self.summary.grid(row=18, column=0, columnspan=2, pady=(12, 0), sticky="nsew")

        # ── right workspace: notebook with tabs ──────────────────────
        workspace = ttk.Frame(self, padding=(0, 12, 12, 12))
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(workspace)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Tab 1 – Heatmap + curvas de nivel
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Mapa de calor / Curvas de nivel")
        tab1.columnconfigure(0, weight=1)
        tab1.rowconfigure(0, weight=1)
        self._fig_heat, self._ax_heat = plt.subplots(figsize=(6, 5), dpi=96)
        self._fig_heat.patch.set_facecolor("#f6f7f8")
        self._canvas_heat = FigureCanvasTkAgg(self._fig_heat, master=tab1)
        self._canvas_heat.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Tab 2 – Evolución de mínimos
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Evolución de mínimos")
        tab2.columnconfigure(0, weight=1)
        tab2.rowconfigure(0, weight=1)
        self._fig_min, self._axes_min = plt.subplots(1, 3, figsize=(10, 4), dpi=96)
        self._fig_min.patch.set_facecolor("#f6f7f8")
        self._canvas_min = FigureCanvasTkAgg(self._fig_min, master=tab2)
        self._canvas_min.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Tab 3 – Convergencia
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Convergencia del solver")
        tab3.columnconfigure(0, weight=1)
        tab3.rowconfigure(0, weight=1)
        self._fig_conv, self._axes_conv = plt.subplots(1, 2, figsize=(10, 4), dpi=96)
        self._fig_conv.patch.set_facecolor("#f6f7f8")
        self._canvas_conv = FigureCanvasTkAgg(self._fig_conv, master=tab3)
        self._canvas_conv.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _entry(self, parent: ttk.Frame, key: str, label: str, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(row=row, column=1, sticky="ew", pady=2)

    def _on_z_scale(self, *_args) -> None:
        val = round(self.vars["z_slice"].get(), 2)
        self.z_label.configure(text=f"{val:.2f} / 10.00")
        self._draw_heat()

    def _on_z0_changed(self, *_args) -> None:
        try:
            raw = self.vars["z0"].get()
        except tk.TclError:
            return
        clamped = round(min(10.0, max(0.2, raw)), 2)
        if raw != clamped:
            self.vars["z0"].set(clamped)
        self.vars["z_slice"].set(clamped)
        self.z_label.configure(text=f"{clamped:.2f} / 10.00")

    def _read_real_source(self) -> FuenteSismica:
        z0 = round(min(10.0, max(0.2, self.vars["z0"].get())), 2)
        return FuenteSismica(
            self.vars["x0"].get(),
            self.vars["y0"].get(),
            z0,
            self.vars["a0"].get(),
        )

    def _read_initial_source(self) -> FuenteSismica:
        return FuenteSismica(
            self.vars["ix"].get(),
            self.vars["iy"].get(),
            self.vars["iz"].get(),
            self.vars["ia"].get(),
        )

    def _write_summary(self, text: str) -> None:
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", text)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _simulate(self) -> None:
        self.fuente_real = self._read_real_source()
        self.simulador.simular_amplitudes(self.red, self.fuente_real)
        self.analisis = None
        self.estados = []
        self._write_summary("Datos simulados con ruido gaussiano alpha=0.05.")
        self._draw_heat()
        self._draw_minima()
        self._draw_convergence()

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
        self._draw_heat()
        self._draw_minima()

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
        self._draw_heat()
        self._draw_convergence()

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

    # ------------------------------------------------------------------
    # Drawing: Tab 1 – Mapa de calor + curvas de nivel
    # ------------------------------------------------------------------

    def _draw_heat(self) -> None:
        ax = self._ax_heat
        ax.cla()

        count = 60
        x_min, x_max = -5.8, 5.8
        y_min, y_max = -5.8, 5.8
        z = self.vars["z_slice"].get()
        a0 = self.vars["a0"].get()

        xs = np.linspace(x_min, x_max, count)
        ys = np.linspace(y_min, y_max, count)
        Z = np.array(
            [[self.modelo.error(self.red, FuenteSismica(x, y, z, a0)) for x in xs] for y in ys]
        )

        # Mapa de calor
        img = ax.imshow(
            Z,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            aspect="equal",
            cmap="plasma",
            interpolation="bilinear",
        )
        self._fig_heat.colorbar(img, ax=ax, label="Error E(x,y,z)")

        # Curvas de nivel
        levels = 14
        cs = ax.contour(xs, ys, Z, levels=levels, colors="white", linewidths=0.7, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2g")

        # Sensores
        sensor_xs = [s.x for s in self.red.sensores]
        sensor_ys = [s.y for s in self.red.sensores]
        ax.scatter(sensor_xs, sensor_ys, marker="^", color="#111827", s=60, zorder=5, label="Sensores")
        for s in self.red.sensores:
            ax.annotate(str(s.sensor_id), (s.x, s.y), textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color="#111827")

        # Fuente real
        ax.scatter(
            [self.fuente_real.x0], [self.fuente_real.y0],
            marker="x", s=120, linewidths=2.5, color="#ef4444", zorder=6, label="Fuente real"
        )

        # Estimacion (si existe)
        if self.estados:
            est = self.estados[-1].fuente
            ax.scatter(
                [est.x0], [est.y0],
                marker="o", s=90, linewidths=2, facecolors="none", edgecolors="#3b82f6",
                zorder=6, label="Estimado"
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"E(x, y, z={z:.2f})  –  A₀={a0:.0f}")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_facecolor("#1e1e2e")

        self._fig_heat.tight_layout()
        self._canvas_heat.draw()

    # ------------------------------------------------------------------
    # Drawing: Tab 2 – Evolución de mínimos
    # ------------------------------------------------------------------

    def _draw_minima(self) -> None:
        for ax in self._axes_min:
            ax.cla()

        if self.analisis is None:
            self._axes_min[1].text(
                0.5, 0.5,
                "Presiona «Analizar E(x,y,z)»\npara ver la evolución de mínimos",
                ha="center", va="center", fontsize=12, color="#6b7280",
                transform=self._axes_min[1].transAxes,
            )
            self._axes_min[1].set_axis_off()
            self._axes_min[0].set_axis_off()
            self._axes_min[2].set_axis_off()
            self._fig_min.tight_layout()
            self._canvas_min.draw()
            return

        minima = self.analisis["minima"]
        zs = [m["z"] for m in minima]
        xs = [m["x"] for m in minima]
        ys = [m["y"] for m in minima]
        errs = [m["error"] for m in minima]
        gmin = self.analisis["global_minimum"]

        colors = "#6366f1"

        ax_x, ax_y, ax_e = self._axes_min

        # x mínimo vs z
        ax_x.plot(zs, xs, color=colors, linewidth=1.6)
        ax_x.axhline(self.fuente_real.x0, color="#ef4444", linestyle="--", linewidth=1.2, label="x real")
        ax_x.axvline(gmin["z"], color="#f59e0b", linestyle=":", linewidth=1.2, label=f"z*={gmin['z']:.2f}")
        ax_x.scatter([gmin["z"]], [gmin["x"]], color="#f59e0b", zorder=5)
        ax_x.set_xlabel("z")
        ax_x.set_ylabel("x mínimo")
        ax_x.set_title("Mínimo en x por corte z")
        ax_x.legend(fontsize=7)
        ax_x.grid(True, alpha=0.3)

        # y mínimo vs z
        ax_y.plot(zs, ys, color="#10b981", linewidth=1.6)
        ax_y.axhline(self.fuente_real.y0, color="#ef4444", linestyle="--", linewidth=1.2, label="y real")
        ax_y.axvline(gmin["z"], color="#f59e0b", linestyle=":", linewidth=1.2, label=f"z*={gmin['z']:.2f}")
        ax_y.scatter([gmin["z"]], [gmin["y"]], color="#f59e0b", zorder=5)
        ax_y.set_xlabel("z")
        ax_y.set_ylabel("y mínimo")
        ax_y.set_title("Mínimo en y por corte z")
        ax_y.legend(fontsize=7)
        ax_y.grid(True, alpha=0.3)

        # error mínimo vs z
        ax_e.plot(zs, errs, color="#f97316", linewidth=1.6)
        ax_e.axvline(gmin["z"], color="#f59e0b", linestyle=":", linewidth=1.2, label=f"z*={gmin['z']:.2f}")
        ax_e.scatter([gmin["z"]], [gmin["error"]], color="#ef4444", zorder=5, label=f"E*={gmin['error']:.3g}")
        ax_e.set_xlabel("z")
        ax_e.set_ylabel("Error mínimo E")
        ax_e.set_title("Error mínimo por corte z")
        ax_e.legend(fontsize=7)
        ax_e.grid(True, alpha=0.3)
        ax_e.set_yscale("log")

        self._fig_min.suptitle(
            f"Evolución de mínimos  |  Fuente real: ({self.fuente_real.x0:.2f}, {self.fuente_real.y0:.2f}, {self.fuente_real.z0:.2f})",
            fontsize=10,
        )
        self._fig_min.tight_layout()
        self._canvas_min.draw()

    # ------------------------------------------------------------------
    # Drawing: Tab 3 – Convergencia del solver
    # ------------------------------------------------------------------

    def _draw_convergence(self) -> None:
        for ax in self._axes_conv:
            ax.cla()

        if not self.estados:
            self._axes_conv[0].text(
                0.5, 0.5,
                "Presiona «Resolver inverso»\npara ver la convergencia",
                ha="center", va="center", fontsize=12, color="#6b7280",
                transform=self._axes_conv[0].transAxes,
            )
            self._axes_conv[0].set_axis_off()
            self._axes_conv[1].set_axis_off()
            self._fig_conv.tight_layout()
            self._canvas_conv.draw()
            return

        iters = [s.iteracion for s in self.estados]
        errors = [s.error for s in self.estados]
        deltas = [s.delta_norm for s in self.estados]

        ax_err, ax_delta = self._axes_conv

        # Error vs iteración
        ax_err.semilogy(iters, errors, marker="o", markersize=4, color="#6366f1", linewidth=1.8)
        ax_err.set_xlabel("Iteración")
        ax_err.set_ylabel("Error  E  (escala log)")
        ax_err.set_title("Convergencia del error")
        ax_err.grid(True, which="both", alpha=0.3)
        ax_err.set_xticks(iters)
        ax_err.annotate(
            f"E final = {errors[-1]:.3g}",
            xy=(iters[-1], errors[-1]),
            xytext=(-60, 12),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#374151"),
            color="#374151",
        )

        # Δ vs iteración (skip iter 0 which has delta=0)
        if len(iters) > 1:
            ax_delta.semilogy(iters[1:], deltas[1:], marker="s", markersize=4, color="#10b981", linewidth=1.8)
        ax_delta.set_xlabel("Iteración")
        ax_delta.set_ylabel("|Δm|  (escala log)")
        ax_delta.set_title("Norma del paso (||Δm||)")
        ax_delta.grid(True, which="both", alpha=0.3)
        if len(iters) > 1:
            ax_delta.set_xticks(iters[1:])

        final = self.estados[-1]
        self._fig_conv.suptitle(
            f"Solver inverso  |  {final.iteracion} iter  |  "
            f"Estimado: ({final.fuente.x0:.3f}, {final.fuente.y0:.3f}, {final.fuente.z0:.3f})",
            fontsize=10,
        )
        self._fig_conv.tight_layout()
        self._canvas_conv.draw()


if __name__ == "__main__":
    SeismicTkApp().mainloop()
