from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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

_X_MIN, _X_MAX = -5.8, 5.8
_Y_MIN, _Y_MAX = -5.8, 5.8
_Z_MIN, _Z_MAX = 0.2, 10.0
_GRID_HEAT = 60   # grid resolution for heatmap
_GRID_STAB = 38   # grid resolution for stability (SVD is heavier)
_ANIM_STEP = 0.18  # z increment per animation frame (km)
_ANIM_MS   = 110   # ms between animation frames

_TAB_HEAT = 0
_TAB_STAB = 1
_TAB_MIN  = 2
_TAB_CONV = 3


class SeismicTkApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Localizacion de fuente sismica")
        self.geometry("1360x840")
        self.minsize(1100, 700)

        self.modelo      = ModeloAtenuacion()
        self.simulador   = SimuladorDatos()
        self.red         = default_sensor_network()
        self.fuente_real = self.simulador.fuente_por_defecto()
        self.estados: list = []
        self.analisis: dict | None = None
        self._anim_id: str | None = None

        self.vars = {
            "x0":      tk.DoubleVar(value=self.fuente_real.x0),
            "y0":      tk.DoubleVar(value=self.fuente_real.y0),
            "z0":      tk.DoubleVar(value=self.fuente_real.z0),
            "a0":      tk.DoubleVar(value=self.fuente_real.a0),
            "ix":      tk.DoubleVar(value=0.0),
            "iy":      tk.DoubleVar(value=0.0),
            "iz":      tk.DoubleVar(value=1.0),
            "ia":      tk.DoubleVar(value=700.0),
            "z_slice": tk.DoubleVar(value=self.fuente_real.z0),
        }

        self._build_ui()
        self.vars["z0"].trace_add("write", self._on_z0_changed)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
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

        row = 0
        ttk.Label(controls, text="Fuente real", font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w"); row += 1
        self._entry(controls, "x0", "x real",  row); row += 1
        self._entry(controls, "y0", "y real",  row); row += 1
        self._entry(controls, "z0", "z real",  row); row += 1
        self._entry(controls, "a0", "A0 real", row); row += 1

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, pady=8, sticky="ew"); row += 1
        ttk.Label(controls, text="Estimacion inicial", font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w"); row += 1
        self._entry(controls, "ix", "x inicial",  row); row += 1
        self._entry(controls, "iy", "y inicial",  row); row += 1
        self._entry(controls, "iz", "z inicial",  row); row += 1
        self._entry(controls, "ia", "A0 inicial", row); row += 1

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, pady=8, sticky="ew"); row += 1
        ttk.Label(controls, text="Corte z", font=("", 9, "bold")).grid(
            row=row, column=0, sticky="w")
        self.z_label = ttk.Label(controls, text=f"{self.vars['z_slice'].get():.2f}")
        self.z_label.grid(row=row, column=1, sticky="e"); row += 1
        z_scale = ttk.Scale(
            controls, from_=_Z_MIN, to=_Z_MAX,
            variable=self.vars["z_slice"], command=self._on_z_scale)
        z_scale.grid(row=row, column=0, columnspan=2, sticky="ew"); row += 1

        self._btn_anim = ttk.Button(controls, text="Animar z", command=self._toggle_anim)
        self._btn_anim.grid(row=row, column=0, columnspan=2, pady=(2, 6), sticky="ew"); row += 1

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, pady=4, sticky="ew"); row += 1
        ttk.Button(controls, text="Simular datos",     command=self._simulate).grid(
            row=row, column=0, columnspan=2, pady=3, sticky="ew"); row += 1
        ttk.Button(controls, text="Analizar E(x,y,z)", command=self._analyze).grid(
            row=row, column=0, columnspan=2, pady=3, sticky="ew"); row += 1
        ttk.Button(controls, text="Resolver inverso",  command=self._solve).grid(
            row=row, column=0, columnspan=2, pady=3, sticky="ew"); row += 1
        ttk.Button(controls, text="Exportar JSON",     command=self._export).grid(
            row=row, column=0, columnspan=2, pady=3, sticky="ew"); row += 1

        self.summary = tk.Text(controls, width=34, height=13, wrap="word")
        self.summary.grid(row=row, column=0, columnspan=2, pady=(8, 0), sticky="nsew")
        controls.rowconfigure(row, weight=1)

        # ── right workspace ─────────────────────────────────────────────
        workspace = ttk.Frame(self, padding=(0, 10, 10, 10))
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(workspace)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Tab 0 – Mapa de calor + curvas de nivel (fixed gridspec, no shrink)
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Mapa de calor / Curvas de nivel")
        tab1.columnconfigure(0, weight=1); tab1.rowconfigure(0, weight=1)
        self._fig_heat = plt.figure(figsize=(7, 5.5), dpi=96)
        self._fig_heat.patch.set_facecolor("#f6f7f8")
        gs1 = self._fig_heat.add_gridspec(
            1, 2, width_ratios=[22, 1],
            left=0.08, right=0.95, bottom=0.10, top=0.91, wspace=0.04)
        self._ax_heat      = self._fig_heat.add_subplot(gs1[0, 0])
        self._ax_cbar_heat = self._fig_heat.add_subplot(gs1[0, 1])
        self._canvas_heat = FigureCanvasTkAgg(self._fig_heat, master=tab1)
        self._canvas_heat.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Tab 1 – Estabilidad Geometrica
        tab_stab = ttk.Frame(self.notebook)
        self.notebook.add(tab_stab, text="Estabilidad Geometrica")
        tab_stab.columnconfigure(0, weight=1); tab_stab.rowconfigure(0, weight=1)
        self._fig_stab = plt.figure(figsize=(10, 4.8), dpi=96)
        self._fig_stab.patch.set_facecolor("#f6f7f8")
        gs2 = self._fig_stab.add_gridspec(
            1, 4, width_ratios=[20, 1, 20, 1],
            left=0.07, right=0.97, bottom=0.12, top=0.88, wspace=0.10)
        self._ax_cond      = self._fig_stab.add_subplot(gs2[0, 0])
        self._ax_cbar_cond = self._fig_stab.add_subplot(gs2[0, 1])
        self._ax_sv        = self._fig_stab.add_subplot(gs2[0, 2])
        self._ax_cbar_sv   = self._fig_stab.add_subplot(gs2[0, 3])
        self._canvas_stab = FigureCanvasTkAgg(self._fig_stab, master=tab_stab)
        self._canvas_stab.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Tab 2 – Evolucion de minimos
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Evolucion de minimos")
        tab2.columnconfigure(0, weight=1); tab2.rowconfigure(0, weight=1)
        self._fig_min = plt.figure(figsize=(10, 4.2), dpi=96)
        self._fig_min.patch.set_facecolor("#f6f7f8")
        self._axes_min = self._fig_min.subplots(1, 3)
        self._fig_min.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.86, wspace=0.35)
        self._canvas_min = FigureCanvasTkAgg(self._fig_min, master=tab2)
        self._canvas_min.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Tab 3 – Convergencia del solver
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Convergencia del solver")
        tab3.columnconfigure(0, weight=1); tab3.rowconfigure(0, weight=1)
        self._fig_conv = plt.figure(figsize=(10, 4.2), dpi=96)
        self._fig_conv.patch.set_facecolor("#f6f7f8")
        self._axes_conv = self._fig_conv.subplots(1, 2)
        self._fig_conv.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.86, wspace=0.35)
        self._canvas_conv = FigureCanvasTkAgg(self._fig_conv, master=tab3)
        self._canvas_conv.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _entry(self, parent, key: str, label: str, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.vars[key], width=12).grid(
            row=row, column=1, sticky="ew", pady=2)

    def _current_tab(self) -> int:
        try:
            return self.notebook.index(self.notebook.select())
        except tk.TclError:
            return _TAB_HEAT

    def _on_z_scale(self, *_args) -> None:
        val = round(self.vars["z_slice"].get(), 2)
        self.z_label.configure(text=f"{val:.2f}")
        self._draw_heat()
        if self._current_tab() == _TAB_STAB:
            self._draw_stability()

    def _on_z0_changed(self, *_args) -> None:
        try:
            raw = self.vars["z0"].get()
        except tk.TclError:
            return
        clamped = round(min(_Z_MAX, max(_Z_MIN, raw)), 2)
        if raw != clamped:
            self.vars["z0"].set(clamped)
        self.vars["z_slice"].set(clamped)
        self.z_label.configure(text=f"{clamped:.2f}")

    def _on_tab_changed(self, *_args) -> None:
        if self._current_tab() == _TAB_STAB:
            self._draw_stability()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _toggle_anim(self) -> None:
        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None
            self._btn_anim.configure(text="Animar z")
        else:
            self._btn_anim.configure(text="Detener")
            self._anim_step()

    def _anim_step(self) -> None:
        z = round(self.vars["z_slice"].get() + _ANIM_STEP, 2)
        if z > _Z_MAX:
            z = _Z_MIN
        self.vars["z_slice"].set(z)
        self.z_label.configure(text=f"{z:.2f}")
        self._draw_heat()
        if self._current_tab() == _TAB_STAB:
            self._draw_stability()
        self._anim_id = self.after(_ANIM_MS, self._anim_step)

    def _stop_anim(self) -> None:
        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None
            self._btn_anim.configure(text="Animar z")

    def _read_real_source(self) -> FuenteSismica:
        z0 = round(min(_Z_MAX, max(_Z_MIN, self.vars["z0"].get())), 2)
        return FuenteSismica(
            self.vars["x0"].get(), self.vars["y0"].get(),
            z0, self.vars["a0"].get())

    def _read_initial_source(self) -> FuenteSismica:
        return FuenteSismica(
            self.vars["ix"].get(), self.vars["iy"].get(),
            self.vars["iz"].get(), self.vars["ia"].get())

    def _compute_Z(self, count: int, z: float, a0: float):
        xs = np.linspace(_X_MIN, _X_MAX, count)
        ys = np.linspace(_Y_MIN, _Y_MAX, count)
        Z = np.array(
            [[self.modelo.error(self.red, FuenteSismica(float(x), float(y), z, a0))
              for x in xs]
             for y in ys])
        return xs, ys, Z

    def _overlay(self, ax, show_estimate: bool = True) -> None:
        """Draw sensors, real source and (optionally) estimated source on ax."""
        for s in self.red.sensores:
            ax.scatter(s.x, s.y, marker="^", color="white", s=55, zorder=5)
            ax.annotate(str(s.sensor_id), (s.x, s.y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="white")
        ax.scatter(
            self.fuente_real.x0, self.fuente_real.y0,
            marker="x", s=130, linewidths=2.5, color="#ef4444", zorder=6, label="Fuente real")
        if show_estimate and self.estados:
            est = self.estados[-1].fuente
            ax.scatter(
                est.x0, est.y0,
                marker="o", s=90, linewidths=2,
                facecolors="none", edgecolors="#38bdf8", zorder=6, label="Estimado")

    def _write_summary(self, text: str) -> None:
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", text)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _simulate(self) -> None:
        self._stop_anim()
        self.fuente_real = self._read_real_source()
        self.simulador.simular_amplitudes(self.red, self.fuente_real)
        self.analisis = None
        self.estados = []
        self._write_summary("Datos simulados con ruido gaussiano alpha=0.05.")
        self._draw_heat()
        self._draw_minima()
        self._draw_convergence()
        if self._current_tab() == _TAB_STAB:
            self._draw_stability()

    def _analyze(self) -> None:
        self._stop_anim()
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
        self._stop_anim()
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
    # Drawing: Tab 0 – Mapa de calor + curvas de nivel
    # KEY FIX: axes are positioned by gridspec (fixed); colorbar drawn
    # into pre-allocated cax – never calls tight_layout, never shrinks.
    # ------------------------------------------------------------------

    def _draw_heat(self) -> None:
        ax  = self._ax_heat
        cax = self._ax_cbar_heat
        z   = round(self.vars["z_slice"].get(), 2)
        a0  = self.vars["a0"].get()

        xs, ys, Z = self._compute_Z(_GRID_HEAT, z, a0)

        # Clear data axes and colorbar axis only; gridspec positions stay fixed
        ax.cla()
        cax.cla()
        ax.set_facecolor("#1a1a2e")

        img = ax.imshow(
            Z,
            extent=[_X_MIN, _X_MAX, _Y_MIN, _Y_MAX],
            origin="lower", aspect="equal",
            cmap="plasma", interpolation="bilinear",
        )
        # colorbar goes into the fixed cax – no axis resizing
        self._fig_heat.colorbar(img, cax=cax, label="E(x,y,z)")

        cs = ax.contour(xs, ys, Z, levels=14, colors="white", linewidths=0.65, alpha=0.75)
        ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.2g")

        self._overlay(ax)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.4)
        ax.set_xlim(_X_MIN, _X_MAX); ax.set_ylim(_Y_MIN, _Y_MAX)
        ax.set_xlabel("x [km]");     ax.set_ylabel("y [km]")
        ax.set_title(f"E(x, y, z = {z:.2f} km)  |  A0 = {a0:.0f}", fontsize=10)

        self._canvas_heat.draw_idle()

    # ------------------------------------------------------------------
    # Drawing: Tab 1 – Estabilidad Geometrica
    # ------------------------------------------------------------------

    def _draw_stability(self) -> None:
        z  = round(self.vars["z_slice"].get(), 2)
        a0 = self.vars["a0"].get()

        xs = np.linspace(_X_MIN, _X_MAX, _GRID_STAB)
        ys = np.linspace(_Y_MIN, _Y_MAX, _GRID_STAB)

        cond_grid   = np.empty((_GRID_STAB, _GRID_STAB))
        sv_min_grid = np.empty((_GRID_STAB, _GRID_STAB))

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                J  = np.array(self.modelo.jacobiano(
                    self.red, FuenteSismica(float(x), float(y), z, a0)))
                sv = np.linalg.svd(J, compute_uv=False)
                cond_grid[yi, xi]   = sv[0] / (sv[-1] + 1e-12)
                sv_min_grid[yi, xi] = sv[-1]

        log_cond = np.log10(np.clip(cond_grid, 1.0, None))
        extent   = [_X_MIN, _X_MAX, _Y_MIN, _Y_MAX]

        # ── condition number ──────────────────────────────────────────
        ax1  = self._ax_cond;  cax1 = self._ax_cbar_cond
        ax1.cla(); cax1.cla()
        ax1.set_facecolor("#1a1a2e")
        img1 = ax1.imshow(log_cond, extent=extent, origin="lower",
                          aspect="equal", cmap="viridis_r", interpolation="bilinear")
        self._fig_stab.colorbar(img1, cax=cax1, label="log10 k(J)")
        cs1 = ax1.contour(xs, ys, log_cond, levels=10,
                          colors="white", linewidths=0.55, alpha=0.65)
        ax1.clabel(cs1, inline=True, fontsize=5.5, fmt="%.1f")
        self._overlay(ax1, show_estimate=False)
        ax1.set_xlim(_X_MIN, _X_MAX); ax1.set_ylim(_Y_MIN, _Y_MAX)
        ax1.set_xlabel("x [km]");     ax1.set_ylabel("y [km]")
        ax1.set_title(
            f"Numero de condicion k(J)  |  z = {z:.2f} km\n"
            "(verde = bien condicionado, menor es mejor)", fontsize=9)

        # ── minimum singular value ────────────────────────────────────
        ax2  = self._ax_sv;  cax2 = self._ax_cbar_sv
        ax2.cla(); cax2.cla()
        ax2.set_facecolor("#1a1a2e")
        img2 = ax2.imshow(sv_min_grid, extent=extent, origin="lower",
                          aspect="equal", cmap="plasma", interpolation="bilinear")
        self._fig_stab.colorbar(img2, cax=cax2, label="sv_min(J)")
        cs2 = ax2.contour(xs, ys, sv_min_grid, levels=10,
                          colors="white", linewidths=0.55, alpha=0.65)
        ax2.clabel(cs2, inline=True, fontsize=5.5, fmt="%.2g")
        self._overlay(ax2, show_estimate=False)
        ax2.set_xlim(_X_MIN, _X_MAX); ax2.set_ylim(_Y_MIN, _Y_MAX)
        ax2.set_xlabel("x [km]");     ax2.set_ylabel("y [km]")
        ax2.set_title(
            f"Valor singular minimo sv_min(J)  |  z = {z:.2f} km\n"
            "(mayor = mas informacion, mejor localizacion)", fontsize=9)

        self._canvas_stab.draw_idle()

    # ------------------------------------------------------------------
    # Drawing: Tab 2 – Evolucion de minimos
    # ------------------------------------------------------------------

    def _draw_minima(self) -> None:
        for ax in self._axes_min:
            ax.cla()

        if self.analisis is None:
            self._axes_min[1].text(
                0.5, 0.5,
                "Presiona Analizar E(x,y,z)\npara ver la evolucion de minimos",
                ha="center", va="center", fontsize=11, color="#6b7280",
                transform=self._axes_min[1].transAxes)
            for ax in self._axes_min:
                ax.set_axis_off()
            self._canvas_min.draw_idle()
            return

        minima = self.analisis["minima"]
        zs   = [m["z"]     for m in minima]
        xs   = [m["x"]     for m in minima]
        ys   = [m["y"]     for m in minima]
        errs = [m["error"] for m in minima]
        gmin = self.analisis["global_minimum"]
        z_now = self.vars["z_slice"].get()

        ax_x, ax_y, ax_e = self._axes_min

        ax_x.plot(zs, xs, color="#818cf8", linewidth=1.7)
        ax_x.axhline(self.fuente_real.x0, color="#ef4444", ls="--", lw=1.2, label="x real")
        ax_x.axvline(gmin["z"], color="#f59e0b", ls=":", lw=1.2, label=f"z*={gmin['z']:.2f}")
        ax_x.axvline(z_now, color="#38bdf8", ls="-", lw=1.0, alpha=0.7, label=f"z={z_now:.2f}")
        ax_x.scatter([gmin["z"]], [gmin["x"]], color="#f59e0b", zorder=5, s=50)
        ax_x.set_xlabel("z [km]"); ax_x.set_ylabel("x minimo")
        ax_x.set_title("Minimo en x por corte z"); ax_x.legend(fontsize=7)
        ax_x.grid(True, alpha=0.25)

        ax_y.plot(zs, ys, color="#34d399", linewidth=1.7)
        ax_y.axhline(self.fuente_real.y0, color="#ef4444", ls="--", lw=1.2, label="y real")
        ax_y.axvline(gmin["z"], color="#f59e0b", ls=":", lw=1.2, label=f"z*={gmin['z']:.2f}")
        ax_y.axvline(z_now, color="#38bdf8", ls="-", lw=1.0, alpha=0.7, label=f"z={z_now:.2f}")
        ax_y.scatter([gmin["z"]], [gmin["y"]], color="#f59e0b", zorder=5, s=50)
        ax_y.set_xlabel("z [km]"); ax_y.set_ylabel("y minimo")
        ax_y.set_title("Minimo en y por corte z"); ax_y.legend(fontsize=7)
        ax_y.grid(True, alpha=0.25)

        ax_e.semilogy(zs, errs, color="#fb923c", linewidth=1.7)
        ax_e.axvline(gmin["z"], color="#f59e0b", ls=":", lw=1.2, label=f"z*={gmin['z']:.2f}")
        ax_e.axvline(z_now, color="#38bdf8", ls="-", lw=1.0, alpha=0.7, label=f"z={z_now:.2f}")
        ax_e.scatter([gmin["z"]], [gmin["error"]], color="#ef4444", zorder=5, s=50,
                     label=f"E*={gmin['error']:.3g}")
        ax_e.set_xlabel("z [km]"); ax_e.set_ylabel("Error minimo E")
        ax_e.set_title("Error minimo por corte z (log)"); ax_e.legend(fontsize=7)
        ax_e.grid(True, which="both", alpha=0.25)

        self._fig_min.suptitle(
            f"Evolucion de minimos  |  Fuente real: "
            f"({self.fuente_real.x0:.2f}, {self.fuente_real.y0:.2f}, {self.fuente_real.z0:.2f})",
            fontsize=10)
        self._canvas_min.draw_idle()

    # ------------------------------------------------------------------
    # Drawing: Tab 3 – Convergencia del solver
    # ------------------------------------------------------------------

    def _draw_convergence(self) -> None:
        for ax in self._axes_conv:
            ax.cla()

        if not self.estados:
            self._axes_conv[0].text(
                0.5, 0.5,
                "Presiona Resolver inverso\npara ver la convergencia",
                ha="center", va="center", fontsize=11, color="#6b7280",
                transform=self._axes_conv[0].transAxes)
            for ax in self._axes_conv:
                ax.set_axis_off()
            self._canvas_conv.draw_idle()
            return

        iters  = [s.iteracion  for s in self.estados]
        errors = [s.error      for s in self.estados]
        deltas = [s.delta_norm for s in self.estados]

        ax_e, ax_d = self._axes_conv

        ax_e.semilogy(iters, errors, "o-", color="#818cf8", markersize=4, linewidth=1.8)
        ax_e.set_xlabel("Iteracion"); ax_e.set_ylabel("Error E (log)")
        ax_e.set_title("Convergencia del error")
        ax_e.set_xticks(iters)
        ax_e.grid(True, which="both", alpha=0.25)
        ax_e.annotate(
            f"E={errors[-1]:.3g}",
            xy=(iters[-1], errors[-1]), xytext=(-55, 14),
            textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#9ca3af"), color="#374151")

        if len(iters) > 1:
            ax_d.semilogy(iters[1:], deltas[1:], "s-", color="#34d399",
                          markersize=4, linewidth=1.8)
        ax_d.set_xlabel("Iteracion"); ax_d.set_ylabel("|Dm| (log)")
        ax_d.set_title("Norma del paso |Dm|")
        if len(iters) > 1:
            ax_d.set_xticks(iters[1:])
        ax_d.grid(True, which="both", alpha=0.25)

        final = self.estados[-1]
        self._fig_conv.suptitle(
            f"Solver inverso  |  {final.iteracion} iter  |  "
            f"Estimado: ({final.fuente.x0:.3f}, {final.fuente.y0:.3f}, {final.fuente.z0:.3f})",
            fontsize=10)
        self._canvas_conv.draw_idle()


if __name__ == "__main__":
    SeismicTkApp().mainloop()
