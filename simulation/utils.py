import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

import config as cfg
from mobility import MobilityModel


def seconds_to_tu(seconds: float, time_unit: cfg.TimeUnit = cfg.TIME_UNIT) -> float:
    if time_unit == cfg.TimeUnit.SECOND:
        return seconds
    if time_unit == cfg.TimeUnit.MILLISECOND:
        return seconds * 1000.0
    raise ValueError("Unknown TIME_UNIT")


def tu_to_seconds(time_unit_value: float, time_unit: cfg.TimeUnit = cfg.TIME_UNIT) -> float:
    if time_unit == cfg.TimeUnit.SECOND:
        return time_unit_value
    if time_unit == cfg.TimeUnit.MILLISECOND:
        return time_unit_value / 1000.0
    raise ValueError("Unknown TIME_UNIT")


def watt_to_wpt(watt: float, time_unit: cfg.TimeUnit = cfg.TIME_UNIT) -> float:
    if time_unit == cfg.TimeUnit.SECOND:
        return watt / 3600.0
    if time_unit == cfg.TimeUnit.MILLISECOND:
        return watt / (3600.0 * 1000.0)
    raise ValueError("Unknown TIME_UNIT")


def ips_to_ipt(ips: float, time_unit: cfg.TimeUnit = cfg.TIME_UNIT) -> float:
    if time_unit == cfg.TimeUnit.SECOND:
        return ips
    if time_unit == cfg.TimeUnit.MILLISECOND:
        return ips * 10 ** -3
    raise ValueError("Unknown TIME_UNIT")


def generate_mobility_animation(
    mobility_model: MobilityModel,
    output_path: Path,
    app_name: str,
    optimization_method: str,
    current_execution_mode: str,
    animation_format: str,
    animation_step_stride: int,
    placement_history_path: Optional[Path] = None,
    simulated_until_step: Optional[int] = None,
):
    """
    Create a lightweight animation for user movement, nearest fog selection,
    and real task processing placement (EDGE/FOG) if history is provided.
    """
    if not mobility_model.user_trace or not mobility_model.fog_devices:
        return

    effective_stride = max(1, animation_step_stride)
    n_steps = len(mobility_model.user_trace)
    fog_lats = np.array([f.latitude for f in mobility_model.fog_devices])
    fog_lons = np.array([f.longitude for f in mobility_model.fog_devices])
    fog_by_topo_id = {f.topo_id: f for f in mobility_model.fog_devices}

    placement_by_step: Dict[int, Dict[str, object]] = {}
    if placement_history_path is not None and placement_history_path.exists():
        df = pd.read_csv(placement_history_path)
        for step, g in df.groupby("sim_time_s", as_index=False):
            row = g.iloc[0]
            placement_by_step[int(step)] = {
                "execution_mode": str(row.get("execution_mode", "EDGE")).upper(),
                "task_processing_topology_id": row.get("task_processing_topology_id", np.nan),
            }
        if simulated_until_step is None and not df.empty:
            simulated_until_step = int(df["sim_time_s"].max())

    if simulated_until_step is not None:
        n_steps = min(n_steps, max(1, int(simulated_until_step) + 1))

    est_frames = n_steps // effective_stride
    if est_frames > cfg.ANIMATION_MAX_FRAMES:
        effective_stride = int(np.ceil(float(n_steps) / float(cfg.ANIMATION_MAX_FRAMES)))
    frames = list(range(0, n_steps, effective_stride))

    fig, ax = plt.subplots(figsize=(8, 8))
    method_tag = optimization_method.upper()
    ax.scatter(fog_lons, fog_lats, s=10, c="lightgray", label=f"City fog nodes ({method_tag})")
    path_lons = np.array([p[1] for p in mobility_model.user_trace])
    path_lats = np.array([p[0] for p in mobility_model.user_trace])
    ax.plot(path_lons, path_lats, linewidth=1.0, color="#2a9d8f", alpha=0.4, label=f"User path ({method_tag})")

    user_point, = ax.plot([], [], "o", color="#e76f51", markersize=8, label=f"User ({method_tag})")
    nearest_point, = ax.plot([], [], "o", color="#264653", markersize=8, label="Nearest fog (distance)")
    active_point, = ax.plot([], [], marker="*", color="#1d3557", markersize=11, linestyle="None", label=f"Active processing node ({method_tag})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")

    def _update(step: int):
        lat, lon = mobility_model.user_position(step)
        nearest_topo = mobility_model.nearest_fog_topology_node(step)
        nearest = next((f for f in mobility_model.fog_devices if f.topo_id == nearest_topo), None)
        user_point.set_data([lon], [lat])
        if nearest is not None:
            nearest_point.set_data([nearest.longitude], [nearest.latitude])

        mode = str(current_execution_mode).upper()
        placement = placement_by_step.get(step)
        if placement is not None:
            mode = str(placement.get("execution_mode", mode)).upper()
            topo_id = placement.get("task_processing_topology_id", np.nan)
            if mode == "FOG" and pd.notna(topo_id):
                fog = fog_by_topo_id.get(int(topo_id))
                if fog is not None:
                    active_point.set_data([fog.longitude], [fog.latitude])
                else:
                    active_point.set_data([], [])
            else:
                active_point.set_data([lon], [lat])
        else:
            if mode == "FOG":
                active_point.set_data([], [])
            else:
                active_point.set_data([lon], [lat])

        ax.set_title(f"{app_name} | {method_tag} | t={step}s | mode={mode}")
        return user_point, nearest_point, active_point

    ani = FuncAnimation(fig, _update, frames=frames, interval=80, blit=False, repeat=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if animation_format.lower() == "gif":
            ani.save(output_path.with_suffix(".gif"), writer=PillowWriter(fps=8))
        else:
            ani.save(output_path.with_suffix(".mp4"), writer=FFMpegWriter(fps=12))
    except Exception as ex:
        logging.warning("Animation export failed (%s). Falling back to final snapshot.", ex)
        lat, lon = mobility_model.user_position(n_steps - 1)
        ax.plot([lon], [lat], "o", color="#e76f51", markersize=8)
        ax.set_title("Final user position")
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def generate_energy_decision_plot(
    decision_history_path: Path,
    output_path: Path,
    app_name: str,
    optimization_method: str,
    decision_period_s: int = cfg.OFFLOADING_DECISION_PERIOD_S,
):
    """
    Plot LP energy terms used for EDGE/FOG decision per simulation step.
    """
    if not decision_history_path.exists():
        logging.warning("Decision history file not found: %s", decision_history_path)
        return

    df = pd.read_csv(decision_history_path)
    required_cols = {"sim_time_s", "energy_edge_wh", "energy_fog_wh"}
    if not required_cols.issubset(set(df.columns)):
        logging.warning("Decision history missing required columns: %s", required_cols)
        return

    plot_mode = "instant"

    sim_t = df["sim_time_s"].astype(float)
    dt_s = sim_t.diff().fillna(float(decision_period_s))
    dt_s = dt_s.where(dt_s > 0.0, float(decision_period_s))
    dt_h = dt_s / 3600.0

    edge_wh = df["real_energy_edge_wh"].astype(float)
    sensors_wh = df["real_energy_sensors_wh"].astype(float)
    total_wh = edge_wh + sensors_wh

    edge_mw = (edge_wh / dt_h) * 1000.0
    sensors_mw = (sensors_wh / dt_h) * 1000.0
    total_mw = (total_wh / dt_h) * 1000.0

    edge_mwh_cum = edge_wh.cumsum() * 1000.0
    sensors_mwh_cum = sensors_wh.cumsum() * 1000.0
    total_mwh_cum = total_wh.cumsum() * 1000.0

    if plot_mode == "both":
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax_inst, ax_int = axes
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax_inst = ax
        ax_int = ax

    if plot_mode in ("instant", "both"):
        ax_inst.plot(sim_t, edge_mw, label="P_edge (mW)", linewidth=1.0, color="#1f77b4")
        ax_inst.set_title(f"{app_name} | {optimization_method.upper()} | Instant Power")
        ax_inst.set_ylabel("Power (mW)")
        ax_inst.grid(True, alpha=0.25)
        ax_inst.legend(loc="upper right")

    if plot_mode in ("integral", "both"):
        ax_int.plot(sim_t, edge_mwh_cum, label="E_edge cum (mWh)", linewidth=1.0, color="#1f77b4")
        ax_int.plot(sim_t, sensors_mwh_cum, label="E_sensors cum (mWh)", linewidth=1.0, color="#ff7f0e")
        ax_int.plot(sim_t, total_mwh_cum, label="E_total cum (mWh)", linewidth=1.1, color="#2ca02c")
        ax_int.set_title(f"{app_name} | {optimization_method.upper()} | Integral Energy")
        ax_int.set_ylabel("Energy (mWh)")
        ax_int.grid(True, alpha=0.25)
        ax_int.legend(loc="upper left")

    if plot_mode == "both":
        axes[-1].set_xlabel("Simulation step (s)")
    else:
        ax.set_xlabel("Simulation step (s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
