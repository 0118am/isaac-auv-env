import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_paths(args):
    if args.csv:
        csv_path = args.csv
        out_path = args.out or os.path.splitext(csv_path)[0] + "_tracking.png"
        return csv_path, out_path

    result_dir = os.path.join(
        "source",
        "results",
        "rsl_rl",
        args.experiment_name,
        args.load_run,
        args.checkpoint[:-3] + "_play",
    )
    csv_path = os.path.join(result_dir, "logs.csv")
    out_path = args.out or os.path.join(result_dir, "tracking_eval.png")
    return csv_path, out_path


def _require_columns(df, columns, csv_path):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing columns in {csv_path}: {missing}\n"
            "This plotter expects logs from play_eval_for_publication_pos.py. "
            "Rerun that eval script after the pandas logging fix."
        )


def _draw_goal_boundaries(axes, steps_per_goal):
    if steps_per_goal <= 0:
        return
    for ax in axes:
        xmax = ax.get_xlim()[1]
        for step in range(steps_per_goal, int(xmax) + 1, steps_per_goal):
            ax.axvline(step, color="0.8", linewidth=0.8, linestyle=":")


def main():
    parser = argparse.ArgumentParser(description="Plot paper-style WarpAUV position/orientation tracking results.")
    parser.add_argument("--csv", type=str, default=None, help="Direct path to a publication position eval logs.csv.")
    parser.add_argument("--load_run", type=str, default=None, help="Run folder name, e.g. 2026-06-08_14-40-56.")
    parser.add_argument("--checkpoint", type=str, default="model_399.pt", help="Checkpoint name used for eval.")
    parser.add_argument("--experiment_name", type=str, default="warpauv_direct", help="Experiment folder name.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path.")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Control timestep in seconds.")
    parser.add_argument("--steps_per_goal", type=int, default=300, help="Number of logged steps per commanded goal.")
    parser.add_argument("--title", type=str, default="WarpAUV Tracking Evaluation", help="Figure title.")
    args = parser.parse_args()

    if not args.csv and not args.load_run:
        raise SystemExit("Provide either --csv or --load_run.")

    csv_path, out_path = _resolve_paths(args)
    df = pd.read_csv(csv_path)

    required = [
        "goal_x",
        "goal_y",
        "goal_z",
        "true_x",
        "true_y",
        "true_z",
        "goal_roll",
        "goal_pitch",
        "goal_yaw",
        "true_roll",
        "true_pitch",
        "true_yaw",
        "pos_error",
        "ang_error",
        "action_cost",
        "total_reward",
    ]
    _require_columns(df, required, csv_path)

    t = np.arange(len(df)) * args.dt
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(args.title)

    colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    for axis_name in ("x", "y", "z"):
        axes[0].plot(t, df[f"true_{axis_name}"], color=colors[axis_name], label=f"{axis_name} actual")
        axes[0].plot(t, df[f"goal_{axis_name}"], color=colors[axis_name], linestyle="--", label=f"{axis_name} goal")
    axes[0].set_ylabel("Position (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    angle_colors = {"roll": "tab:red", "pitch": "tab:purple", "yaw": "tab:brown"}
    for angle in ("roll", "pitch", "yaw"):
        axes[1].plot(t, np.rad2deg(df[f"true_{angle}"]), color=angle_colors[angle], label=f"{angle} actual")
        axes[1].plot(t, np.rad2deg(df[f"goal_{angle}"]), color=angle_colors[angle], linestyle="--", label=f"{angle} goal")
    axes[1].set_ylabel("Attitude (deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(t, df["pos_error"], label="position error (m)", color="tab:blue")
    axes[2].plot(t, np.rad2deg(df["ang_error"]), label="attitude error (deg)", color="tab:red")
    axes[2].set_ylabel("Error")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    axes[3].plot(t, df["action_cost"], label="action cost", color="tab:gray")
    axes[3].plot(t, df["total_reward"], label="total reward", color="tab:green")
    axes[3].set_ylabel("Cost / Reward")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8)

    _draw_goal_boundaries(axes, args.steps_per_goal)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)

    print(f"Saved plot: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Mean pos_error: {df['pos_error'].mean():.4f}; final: {df['pos_error'].iloc[-1]:.4f}")
    print(f"Mean ang_error(rad): {df['ang_error'].mean():.4f}; final: {df['ang_error'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
