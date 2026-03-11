#!/usr/bin/env python3

import argparse
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

# plt.rcParams["font.family"] = "monospace"
# plt.rcParams["font.monospace"] = ["Iosevka NFM"]
plt.rcParams["font.size"] = 8

# Constants
G = 9.81  # Acceleration due to gravity (m/s^2)
KMH_TO_MS = 1 / 3.6
MS_TO_KMH = 3.6


@dataclass
class Train:
    name: str
    mass: float
    adh_mass: float
    davis_a: float
    davis_b: float
    davis_c: float
    rotational_inertia: float
    braking_curve_table: List[Tuple[float, float]]
    tractive_effort_table: List[Tuple[float, float]]

    @classmethod
    def from_yaml(cls, filepath: str) -> "Train":
        """Constructor to create a Train directly from a YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        braking_curve_table = []
        for v, a in data.get("braking_curve_table", []):
            braking_curve_table.append((v * KMH_TO_MS, a))

        tractive_effort_table = []
        for v, f in data.get("tractive_effort_table", []):
            tractive_effort_table.append((v * KMH_TO_MS, f))

        return cls(
            name=data["name"],
            mass=data["mass"],
            adh_mass=data["adh_mass"],
            davis_a=data["davis_a"],
            davis_b=data["davis_b"],
            davis_c=data["davis_c"],
            rotational_inertia=data["rotational_inertia"],
            braking_curve_table=braking_curve_table,
            tractive_effort_table=tractive_effort_table,
        )


@dataclass
class Path:
    positions: np.ndarray
    speed_limits: np.ndarray
    slopes: np.ndarray
    curves: np.ndarray
    station_masks: np.ndarray
    station_pos: np.ndarray
    station_names: List[str]
    dwell_times: np.ndarray

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        dwell_time: float = 0.0,
        ppkm: int = 1000,
        is_reversed: bool = False,
    ) -> "Path":
        """Constructor to create a Path directly from a CSV file."""
        df = pd.read_csv(filepath)

        # Column processing
        if len(df.columns) != 5:
            raise ValueError(
                "CSV must contain 5 columns: pk_m, vel_kmh, curve_m, slope_‰, station_name"
            )

        df.columns = ["x_start", "v_limit", "curve", "slope", "station"]
        df["v_limit"] = df["v_limit"] * KMH_TO_MS

        # File Validation
        for i in range(len(df) - 1):
            if df["x_start"].iloc[i] >= df["x_start"].iloc[i + 1]:
                raise ValueError(
                    f"Rows {i + 1},{i + 2}: segment end must be strictly greater than segment start."
                )
            if df["v_limit"].iloc[i] < 0:
                raise ValueError(f"Row {i + 1}: speed limit cannot be negative.")

        pos_segs, vlim_segs, curve_segs, slope_segs = [], [], [], []
        is_station_segs, dwell_time_segs = [], []
        station_pos, station_names = [], []

        for i in range(len(df) - 1):
            xi = float(df["x_start"].iloc[i])
            xf = float(df["x_start"].iloc[i + 1])
            vlim = float(df["v_limit"].iloc[i])
            curve = float(df["curve"].iloc[i])
            slope = float(df["slope"].iloc[i])
            station = df["station"].iloc[i]

            is_station = (
                pd.notna(station)
                and str(station).strip().lower() != "nan"
                and str(station).strip() != ""
            )
            if is_station:
                station_pos.append(xi)
                station_names.append(str(station))

            dx = xf - xi
            ppm = ppkm / 1000.0
            n_points = math.ceil(dx * ppm)
            pos = np.linspace(xi, xf, n_points + 1)[:-1]

            pos_segs.append(pos)
            vlim_segs.append(np.full(n_points, vlim))
            curve_segs.append(np.full(n_points, curve))
            slope_segs.append(np.full(n_points, slope))
            is_station_segs.append(np.full(n_points, is_station))

            if is_station:
                dwell_time_segs.append(np.full(n_points, dwell_time / n_points))
            else:
                dwell_time_segs.append(np.full(n_points, 0.0))

        # Flatten arrays
        positions = np.concatenate(pos_segs)
        speed_limits = np.concatenate(vlim_segs)
        curves = np.concatenate(curve_segs)
        slopes = np.concatenate(slope_segs)
        station_masks = np.concatenate(is_station_segs)
        dwell_times = np.concatenate(dwell_time_segs)

        # Append last point
        positions = np.append(positions, df["x_start"].iloc[-1])
        speed_limits = np.append(speed_limits, df["v_limit"].iloc[-1])
        curves = np.append(curves, df["curve"].iloc[-1])
        slopes = np.append(slopes, df["slope"].iloc[-1])
        station_masks = np.append(station_masks, True)
        station_pos.append(df["x_start"].iloc[-1])
        station_names.append(str(df["station"].iloc[-1]))
        dwell_times = np.append(dwell_times, dwell_time)

        if is_reversed:
            pos_offset = positions[-1] + positions[0]
            positions = pos_offset - positions[::-1]
            speed_limits = speed_limits[::-1]
            slopes = -slopes[::-1]
            curves = -curves[::-1]
            station_masks = station_masks[::-1]
            dwell_times = dwell_times[::-1]
            station_pos = pos_offset - np.array(station_pos)[::-1]
            station_names = station_names[::-1]

        return cls(
            positions,
            speed_limits,
            slopes,
            curves,
            station_masks,
            np.array(station_pos),
            station_names,
            dwell_times,
        )


@dataclass
class TrainRunSimulation:
    train: Train
    path: Path
    profile: pd.DataFrame


# Physics Functions
def effective_mass(t: Train) -> float:
    """Accounts for the rotational inertia."""
    return t.mass * (1 + 0.01 * t.rotational_inertia)


def rolling_resistance(t: Train, v: float) -> float:
    """Davis equation for baseline train resistance."""
    return (
        t.mass
        * G
        * (t.davis_a + v * MS_TO_KMH * (t.davis_b + t.davis_c * v * MS_TO_KMH))
        * 0.001
    )


def slope_resistance(p: Path, mass: float, index: int) -> float:
    """Resistance due to vertical gradients."""
    return mass * G * p.slopes[index] * 0.001


def curve_resistance(p: Path, mass: float, index: int) -> float:
    """Resistance due to horizontal gradients."""
    curv_res = 0.0
    curv = abs(p.curves[index])
    if curv > 0:
        if curv >= 300:
            denom = max(1.0, curv - 55)
            specific_resistance = 650 / denom
        else:
            denom = max(1.0, curv - 30)
            specific_resistance = 500 / denom
        curv_res = mass * G * specific_resistance * 0.001
    return curv_res


def max_braking_effort(t: Train, v: float) -> float:
    table = t.braking_curve_table
    if v >= table[-1][0]:
        deceleration = table[-1][1]
    else:
        # Find first row where v < row[0]
        idx = next((i for i, row in enumerate(table) if v < row[0]), None)
        deceleration = table[idx][1] if idx is not None else table[0][1]

    braking_force = t.mass * deceleration
    friction_coeff = 0.161 + 2.1 / (v + 12.2)
    weather_coeff = 1.25
    adhesion_limit = t.adh_mass * G * friction_coeff * weather_coeff
    return min(braking_force, adhesion_limit)


def max_tractive_effort(t: Train, v: float) -> float:
    table = t.tractive_effort_table
    if v >= table[-1][0]:
        traction_force = 0.0
    elif v <= table[0][0]:
        traction_force = table[0][1]
    else:
        idx = next((i for i, row in enumerate(table) if v < row[0]), None)
        if idx is not None:
            v1, f1 = table[idx - 1]
            v2, f2 = table[idx]
            p1, p2 = f1 * v1, f2 * v2
            p = p1 + (p2 - p1) / (v2 - v1) * (v - v1)
            traction_force = p / v
        else:
            traction_force = 0.0

    friction_coeff = 0.161 + 2.1 / (v + 12.2)
    weather_coeff = 1.25
    adhesion_limit = t.adh_mass * G * friction_coeff * weather_coeff
    return min(traction_force, adhesion_limit)


def forward_pass(train: Train, path: Path) -> np.ndarray:
    n = len(path.positions)
    v_fwd = np.zeros(n)
    v_train_lim = train.tractive_effort_table[-1][0]

    for i in range(n - 1):
        v_curr = v_fwd[i]
        F_t = max_tractive_effort(train, v_curr)
        R_roll = rolling_resistance(train, v_curr)
        R_slope = slope_resistance(path, train.mass, i)
        R_curve = curve_resistance(path, train.mass, i)

        F_net = F_t - R_roll - R_slope - R_curve
        a = F_net / effective_mass(train)

        ds = path.positions[i + 1] - path.positions[i]
        v_next_sq = v_curr**2 + 2 * a * ds
        v_next = math.sqrt(v_next_sq) if v_next_sq > 0 else 0.0
        v_track_lim = 0.0 if path.station_masks[i + 1] else path.speed_limits[i + 1]

        v_fwd[i + 1] = min(v_next, v_track_lim, v_train_lim)
    return v_fwd


def backward_pass(train: Train, path: Path) -> np.ndarray:
    n = len(path.positions)
    v_bwd = np.zeros(n)
    v_train_lim = train.tractive_effort_table[-1][0]

    for i in range(n - 1, 0, -1):
        v_curr = v_bwd[i]
        F_b = max_braking_effort(train, v_curr)
        R_roll = rolling_resistance(train, v_curr)
        R_slope = slope_resistance(path, train.mass, i)
        R_curve = curve_resistance(path, train.mass, i)

        F_net = F_b + R_roll - R_slope + R_curve
        a = F_net / effective_mass(train)

        ds = path.positions[i] - path.positions[i - 1]
        v_prev_sq = v_curr**2 + 2 * a * ds
        v_prev = math.sqrt(v_prev_sq) if v_prev_sq > 0 else 0.0
        v_track_lim = 0.0 if path.station_masks[i - 1] else path.speed_limits[i - 1]

        v_bwd[i - 1] = min(v_prev, v_track_lim, v_train_lim)
    return v_bwd


def run_simulation(train: Train, path: Path) -> TrainRunSimulation:
    v_fwd = forward_pass(train, path)
    v_bwd = backward_pass(train, path)
    v_final = np.minimum(v_fwd, v_bwd)

    n = len(path.positions)
    time = np.zeros(n)
    energy = np.zeros(n)
    forces = np.zeros(n)

    for i in range(n - 1):
        v1, v2 = v_final[i], v_final[i + 1]
        v_avg = (v1 + v2) / 2.0
        ds = path.positions[i + 1] - path.positions[i]

        dt = ds / v_avg if v_avg > 0 else 0.0
        time[i + 1] = time[i] + dt + path.dwell_times[i]

        acceleration = (v2**2 - v1**2) / (2 * ds) if ds > 0 else 0.0
        R_roll = rolling_resistance(train, v_avg)
        R_slope = slope_resistance(path, train.mass, i)
        R_curve = curve_resistance(path, train.mass, i)

        force = effective_mass(train) * acceleration + R_roll + R_slope + R_curve

        if force >= 0:
            forces[i] = min(force, max_tractive_effort(train, v_avg))
            energy[i + 1] = energy[i] + (forces[i] * ds)
        else:
            forces[i] = max(force, -max_braking_effort(train, v_avg))
            energy[i + 1] = energy[i]

    time[-1] += path.dwell_times[-1]

    profile_df = pd.DataFrame(
        {
            "distance_km": path.positions * 0.001,
            "velocity_kmh": v_final * MS_TO_KMH,
            "time_min": time / 60,
            "force_kN": forces / 1e3,
            "energy_MJ": energy / 1e6,
        }
    )

    return TrainRunSimulation(train, path, profile_df)


def plot_multi_profile(
    profile: pd.DataFrame,
    path: Path,
    train: Train,
    out_filepath: Optional[str] = None,
    show: bool = True,
):
    if not show and out_filepath is None:
        return

    dist_km = profile["distance_km"].values
    speeds_kmh = profile["velocity_kmh"].values
    energy_MJ = profile["energy_MJ"].values
    times_min = profile["time_min"].values

    slopes = path.slopes
    curves = path.curves
    station_pos_km = path.station_pos * 0.001
    speed_limits_kmh = path.speed_limits * MS_TO_KMH

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    ax1, ax2, ax3, ax4, ax5 = axes

    # Inclination profile
    ax1.fill_between(dist_km, 0, np.maximum(slopes, 0), color="red", alpha=0.3)
    ax1.fill_between(dist_km, np.minimum(slopes, 0), 0, color="green", alpha=0.3)
    ax1.step(dist_km, slopes, where="post", color="black", linewidth=1)
    ax1.set_ylabel("Slope [‰]")

    # Curvature profile
    curv_inv = np.array([0.0 if r == 0.0 else 1.0 / r for r in curves])
    ax2.fill_between(dist_km, 0, curv_inv, color="red", alpha=0.3)
    ax2.step(dist_km, curv_inv, where="post", color="black", linewidth=1)
    ax2.set_ylabel("Curv. [1/m]")

    # Speed profile
    ax3.plot(dist_km, speeds_kmh, color="black", linewidth=1.5)
    ax3.step(
        dist_km,
        speed_limits_kmh,
        where="post",
        color="red",
        alpha=0.6,
        linestyle="--",
        linewidth=1.0,
    )
    ax3.set_ylabel("Vel. [km/h]")

    # Energy profile
    ax4.plot(dist_km, energy_MJ, color="black", linewidth=1.5)
    ax4.set_ylabel("Energy [MJ]")

    # Time profile
    ax5.plot(dist_km, times_min, color="black", linewidth=1.5)
    ax5.set_ylabel("Time [min]")
    ax5.set_xlabel("Distance [km]")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.2)

    for ax in axes[:-1]:
        ax.tick_params(bottom=False)

    # Add vertical station lines across all subplots & rotated names to the bottom plot
    y_min, y_max = ax5.get_ylim()
    for name, pos in zip(path.station_names, station_pos_km):
        for ax in axes:
            ax.axvline(x=pos, color="darkblue", linestyle="--", alpha=0.2, linewidth=1)

        # Place the text rotated 90 degrees
        ax5.text(
            pos,
            y_min + (y_max - y_min) * 0.05,
            f"{name} ({pos:0.2f})",
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=6,
            fontweight="bold",
            color="darkblue",
        )

    total_time = profile["time_min"].iloc[-1]
    minutes = int(total_time)
    seconds = int(round((total_time * 60) % 60))
    fig.suptitle(
        f"Train Type: {train.name} | Running Time: {minutes} min {seconds} sec",
        fontweight="bold",
    )
    plt.tight_layout(h_pad=0.5)

    if out_filepath:
        plt.savefig(out_filepath, dpi=200)
        print(f"Plot saved to: {out_filepath}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="TrainRun.py - Train Run Simulation Engine"
    )
    parser.add_argument(
        "--geometry",
        "-g",
        type=str,
        required=True,
        help="Input CSV track geometry file",
    )
    parser.add_argument(
        "--train",
        "-t",
        type=str,
        required=True,
        help="Input YAML file containing rolling stock",
    )
    parser.add_argument(
        "--reverse",
        "-r",
        action="store_true",
        help="Process the track direction in reverse",
    )
    parser.add_argument(
        "--ppkm",
        "-p",
        type=int,
        default=1000,
        help="Number of discretization points per kilometer",
    )
    parser.add_argument(
        "--dwell",
        "-d",
        type=float,
        default=0.0,
        help="Dwell time at stations in seconds",
    )
    parser.add_argument("--out", "-o", type=str, help="Outputs the plot to a file")
    parser.add_argument("--show", "-s", action="store_true", help="Show plots")

    args = parser.parse_args()

    print(f"Loading track geometry data from: {args.geometry}")
    path = Path.from_csv(
        args.geometry, dwell_time=args.dwell, ppkm=args.ppkm, is_reversed=args.reverse
    )

    print(f"Loading train data from: {args.train}")
    train = Train.from_yaml(args.train)

    print("Running simulation...")
    sim = run_simulation(train, path)
    profile = sim.profile

    total_time = profile["time_min"].iloc[-1]
    minutes = int(total_time)
    seconds = int(round((total_time * 60) % 60))
    total_energy = profile["energy_MJ"].iloc[-1]

    print(f"Simulation Complete: {train.name}")
    print(f"Total Running Time: {minutes} min {seconds} sec")
    print(f"Total Energy Consumed: {round(total_energy, 2)} MJ")

    plot_multi_profile(profile, path, train, out_filepath=args.out, show=args.show)


if __name__ == "__main__":
    main()
