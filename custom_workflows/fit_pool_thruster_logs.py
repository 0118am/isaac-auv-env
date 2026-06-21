"""Fit thruster lookup and response profile updates from validated pool CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration_tools import (  # noqa: E402
    fit_battery_voltage_sag,
    fit_thruster_first_order_response,
    fit_thruster_inflow_lookup_table,
    fit_thruster_static_lookup_table,
    fit_thruster_voltage_exponent,
)
from pool_dynamics_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolProfileAuditOptions,
    pool_profile_calibration_log_schemas,
    validate_pool_calibration_log_directory,
)


THRUSTER_LOG_FILENAMES = (
    "thruster_static_stand.csv",
    "thruster_inflow_stand.csv",
    "thruster_step_response.csv",
    "battery_voltage_thrust_samples.csv",
)


@dataclass(frozen=True)
class ThrusterCalibrationPipelineResult:
    cfg_updates: dict[str, Any]
    diagnostics: dict[str, Any]
    source_files: tuple[str, ...]

    def update_payload(self) -> dict[str, Any]:
        return {
            "cfg_updates": self.cfg_updates,
            "domain_randomization_updates": {},
        }

    def report_dict(self) -> dict[str, Any]:
        return {
            "source_files": list(self.source_files),
            "cfg_updates": self.cfg_updates,
            "diagnostics": self.diagnostics,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit thruster lookup and first-order response updates from calibration CSV logs.",
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing thruster_*.csv calibration logs.")
    parser.add_argument("--output", type=Path, required=True, help="Output builder-compatible updates JSON path.")
    parser.add_argument("--report", type=Path, help="Optional detailed fit diagnostics JSON path.")
    parser.add_argument(
        "--physics-dt",
        type=float,
        help="Physics timestep used to convert fitted response delay seconds into command delay steps.",
    )
    parser.add_argument(
        "--deadband-thrust-threshold",
        type=float,
        default=0.05,
        help="Absolute thrust threshold used to estimate command deadband.",
    )
    parser.add_argument(
        "--delay-candidates",
        type=int,
        default=128,
        help="Number of response-delay candidates evaluated for step-response fitting.",
    )
    parser.add_argument("--nominal-voltage", type=float, default=16.0, help="Nominal voltage for thrust scaling.")
    return parser


def fit_thruster_calibration_logs(
    log_dir: Path,
    *,
    physics_dt_s: float | None = None,
    deadband_thrust_threshold: float = 0.05,
    delay_candidate_count: int = 128,
    nominal_voltage: float = 16.0,
) -> ThrusterCalibrationPipelineResult:
    if physics_dt_s is not None and float(physics_dt_s) <= 0.0:
        raise ValueError("physics_dt_s must be positive when provided.")
    if float(deadband_thrust_threshold) < 0.0:
        raise ValueError("deadband_thrust_threshold must be non-negative.")
    if int(delay_candidate_count) != delay_candidate_count or int(delay_candidate_count) < 1:
        raise ValueError("delay_candidate_count must be a positive integer.")
    if float(nominal_voltage) <= 0.0:
        raise ValueError("nominal_voltage must be positive.")

    schemas = pool_profile_calibration_log_schemas(
        NOMINAL_POOL_DYNAMICS_PROFILE,
        PoolProfileAuditOptions(domain_randomization_expected=False),
    )
    schema_by_filename = {schema.filename: schema for schema in schemas}
    source_files = tuple(filename for filename in THRUSTER_LOG_FILENAMES if (log_dir / filename).is_file())
    if not source_files:
        raise ValueError(f"No supported thruster calibration logs found in {log_dir}.")

    source_schemas = tuple(schema_by_filename[filename] for filename in source_files)
    validation = validate_pool_calibration_log_directory(log_dir, source_schemas)
    if not validation.is_valid:
        messages = "; ".join(
            f"{issue.filename}:{issue.row_number or '-'}:{issue.column or '-'} {issue.message}"
            for issue in validation.issues
            if issue.severity == "error"
        )
        raise ValueError(f"Thruster calibration log validation failed: {messages}")

    cfg_updates: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"validation": validation.to_dict()}

    static_path = log_dir / "thruster_static_stand.csv"
    if static_path.is_file():
        rows = _read_csv_rows(static_path)
        labels, grouped_rows = _ordered_thruster_groups(rows)
        command_axis = _sorted_unique_float(row["command"] for row in rows)
        thrust_tables: list[torch.Tensor] = []
        residual_tables: list[torch.Tensor] = []
        count_tables: list[torch.Tensor] = []
        deadbands: list[float] = []
        for group_rows in grouped_rows:
            fit = fit_thruster_static_lookup_table(
                _float_column(group_rows, "command"),
                _float_column(group_rows, "thrust_n"),
                command_points=command_axis,
                deadband_thrust_threshold=deadband_thrust_threshold,
            )
            thrust_tables.append(fit.thrust_points[0])
            residual_tables.append(fit.residual_rms[0])
            count_tables.append(fit.sample_count[0])
            deadbands.append(fit.estimated_deadband)

        combined_thrust = torch.stack(thrust_tables)
        cfg_updates.update(
            {
                "use_thruster_lookup_table": True,
                "thruster_lookup_commands": command_axis,
                "thruster_lookup_thrusts": _shared_or_per_thruster_table(combined_thrust),
                "thruster_deadband": max(deadbands),
            }
        )
        diagnostics["static_lookup"] = {
            "thruster_labels": labels,
            "command_points": command_axis,
            "thrust_points_n": combined_thrust.detach().cpu().tolist(),
            "residual_rms_n": torch.stack(residual_tables).detach().cpu().tolist(),
            "sample_count": torch.stack(count_tables).detach().cpu().tolist(),
            "estimated_deadband_by_thruster": deadbands,
        }

    inflow_path = log_dir / "thruster_inflow_stand.csv"
    if inflow_path.is_file():
        rows = _read_csv_rows(inflow_path)
        labels, grouped_rows = _ordered_thruster_groups(rows)
        command_axis = _sorted_unique_float(row["command"] for row in rows)
        inflow_axis = _sorted_unique_float(row["axial_inflow_speed_mps"] for row in rows)
        thrust_surfaces: list[torch.Tensor] = []
        residual_surfaces: list[torch.Tensor] = []
        count_surfaces: list[torch.Tensor] = []
        for group_rows in grouped_rows:
            fit = fit_thruster_inflow_lookup_table(
                _float_column(group_rows, "command"),
                _float_column(group_rows, "axial_inflow_speed_mps"),
                _float_column(group_rows, "thrust_n"),
                command_points=command_axis,
                inflow_speed_points=inflow_axis,
            )
            thrust_surfaces.append(fit.thrust_points[0])
            residual_surfaces.append(fit.residual_rms[0])
            count_surfaces.append(fit.sample_count[0])

        combined_surface = torch.stack(thrust_surfaces)
        cfg_updates.update(
            {
                "use_thruster_inflow_lookup_table": True,
                "thruster_inflow_lookup_commands": command_axis,
                "thruster_inflow_lookup_speeds": inflow_axis,
                "thruster_inflow_lookup_thrusts": _shared_or_per_thruster_table(combined_surface),
            }
        )
        diagnostics["inflow_lookup"] = {
            "thruster_labels": labels,
            "command_points": command_axis,
            "inflow_speed_points_mps": inflow_axis,
            "thrust_points_n": combined_surface.detach().cpu().tolist(),
            "residual_rms_n": torch.stack(residual_surfaces).detach().cpu().tolist(),
            "sample_count": torch.stack(count_surfaces).detach().cpu().tolist(),
        }

    response_path = log_dir / "thruster_step_response.csv"
    if response_path.is_file():
        rows = _read_csv_rows(response_path)
        time_s = _float_column(rows, "time_s")
        commands = _float_column(rows, "command")
        measured_thrust = _float_column(rows, "measured_thrust_n")
        step_time = _infer_step_time(time_s, commands)
        fit = fit_thruster_first_order_response(
            time_s,
            measured_thrust,
            command_step_time_s=step_time,
            delay_candidate_count=delay_candidate_count,
        )
        cfg_updates.update(fit.to_cfg_updates(physics_dt_s=physics_dt_s))
        diagnostics["first_order_response"] = {
            "command_step_time_s": step_time,
            "time_constant_s": fit.time_constant_s,
            "response_delay_s": fit.response_delay_s,
            "initial_thrust_n": fit.initial_thrust,
            "steady_state_thrust_n": fit.steady_state_thrust,
            "residual_rms_n": fit.residual_rms,
            "sample_count": fit.sample_count,
            "physics_dt_s": physics_dt_s,
            "command_delay_steps": cfg_updates.get("thruster_command_delay_steps"),
        }

    battery_path = log_dir / "battery_voltage_thrust_samples.csv"
    if battery_path.is_file():
        rows = _read_csv_rows(battery_path)
        sag_fit = fit_battery_voltage_sag(
            _float_column(rows, "time_s"),
            _float_column(rows, "voltage_v"),
        )
        cfg_updates.update(sag_fit.to_cfg_updates())
        diagnostics["battery_voltage_sag"] = {
            "initial_voltage_v": sag_fit.initial_voltage,
            "min_observed_voltage_v": sag_fit.min_observed_voltage,
            "voltage_drop_per_s": sag_fit.voltage_drop_per_s,
            "residual_rms_v": sag_fit.residual_rms,
            "sample_count": sag_fit.sample_count,
            "time_origin_s": sag_fit.time_origin_s,
        }
        scale_rows = _rows_with_value(rows, "thrust_scale")
        if scale_rows:
            exponent_fit = fit_thruster_voltage_exponent(
                _float_column(scale_rows, "voltage_v"),
                _float_column(scale_rows, "thrust_scale"),
                nominal_voltage=nominal_voltage,
            )
            cfg_updates.update(exponent_fit.to_cfg_updates())
            diagnostics["battery_thrust_scaling"] = {
                "nominal_voltage_v": exponent_fit.nominal_voltage,
                "thrust_exponent": exponent_fit.thrust_exponent,
                "residual_rms": exponent_fit.residual_rms,
                "sample_count": exponent_fit.sample_count,
            }

    return ThrusterCalibrationPipelineResult(cfg_updates, diagnostics, source_files)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return rows


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _rows_with_value(rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get(name, "").strip()]


def _sorted_unique_float(values) -> list[float]:
    result = sorted({float(value) for value in values})
    if len(result) < 2:
        raise ValueError("Lookup calibration requires at least two distinct axis values.")
    return result


def _ordered_thruster_groups(rows: list[dict[str, str]]) -> tuple[list[str], list[list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["thruster_index"], []).append(row)
    if len(grouped) == 1:
        labels = list(grouped)
    elif len(grouped) == 8:
        expected = {str(index) for index in range(8)}
        if set(grouped) != expected:
            raise ValueError("Eight-curve thruster logs must use thruster_index labels 0 through 7.")
        labels = [str(index) for index in range(8)]
    else:
        raise ValueError("Thruster lookup logs must contain one shared curve or exactly eight curves labeled 0 through 7.")
    return labels, [grouped[label] for label in labels]


def _shared_or_per_thruster_table(table: torch.Tensor) -> list[Any]:
    if table.shape[0] == 1:
        return table[0].detach().cpu().tolist()
    return table.detach().cpu().tolist()


def _infer_step_time(time_s: list[float], commands: list[float], tolerance: float = 1.0e-6) -> float:
    if len(time_s) != len(commands) or len(time_s) < 2:
        raise ValueError("Step-response time and command arrays must have matching length >= 2.")
    baseline = commands[0]
    for time_value, command in zip(time_s[1:], commands[1:]):
        if abs(command - baseline) > float(tolerance):
            return float(time_value)
    raise ValueError("Could not infer a command step from thruster_step_response.csv.")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = fit_thruster_calibration_logs(
            args.log_dir,
            physics_dt_s=args.physics_dt,
            deadband_thrust_threshold=args.deadband_thrust_threshold,
            delay_candidate_count=args.delay_candidates,
            nominal_voltage=args.nominal_voltage,
        )
        _write_json(args.output, result.update_payload())
        if args.report is not None:
            _write_json(args.report, result.report_dict())
    except Exception as exc:
        print(f"Failed to fit thruster calibration logs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
