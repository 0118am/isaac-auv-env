"""Build a measured PoolDynamicsProfile JSON from calibration update files."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pool_dynamics_profile import (  # noqa: E402
    NOMINAL_POOL_DYNAMICS_PROFILE,
    PoolDynamicsProfile,
    load_pool_dynamics_profile_json,
    merge_pool_dynamics_cfg_updates,
    write_pool_dynamics_profile_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge calibration to_cfg_updates() JSON files into a PoolDynamicsProfile JSON.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        help="Optional existing PoolDynamicsProfile JSON to update. Defaults to the nominal profile.",
    )
    parser.add_argument(
        "--updates",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON file containing flat cfg updates, or a wrapper with cfg_updates and "
            "domain_randomization_updates. May be repeated; later files override earlier ones."
        ),
    )
    parser.add_argument(
        "--domain-randomization-updates",
        type=Path,
        action="append",
        default=[],
        help="JSON file containing flat DomainRandomizationProfile updates. May be repeated.",
    )
    parser.add_argument("--name", help="Override the output profile name.")
    parser.add_argument("--description", help="Override the output profile description.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the merged profile JSON.")
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Ignore unknown update keys instead of failing.",
    )
    return parser


def load_update_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load either a flat cfg update mapping or a cfg/domain wrapper mapping."""

    data = _load_json_mapping(path)
    if "cfg_updates" not in data and "domain_randomization_updates" not in data:
        return dict(data), {}

    allowed = {"cfg_updates", "domain_randomization_updates"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown wrapped update field(s): {', '.join(unknown)}.")

    cfg_updates = data.get("cfg_updates", {})
    domain_updates = data.get("domain_randomization_updates", {})
    if not isinstance(cfg_updates, Mapping):
        raise TypeError(f"{path}: cfg_updates must be a mapping.")
    if not isinstance(domain_updates, Mapping):
        raise TypeError(f"{path}: domain_randomization_updates must be a mapping.")
    return dict(cfg_updates), dict(domain_updates)


def build_profile_from_files(
    *,
    base_profile_path: Path | None,
    update_paths: list[Path],
    domain_randomization_update_paths: list[Path] | None = None,
    name: str | None = None,
    description: str | None = None,
    strict: bool = True,
) -> PoolDynamicsProfile:
    base_profile = (
        load_pool_dynamics_profile_json(base_profile_path)
        if base_profile_path is not None
        else NOMINAL_POOL_DYNAMICS_PROFILE
    )
    cfg_updates: list[dict[str, Any]] = []
    domain_updates: list[dict[str, Any]] = []

    for path in update_paths:
        cfg_update, domain_update = load_update_payload(path)
        cfg_updates.append(cfg_update)
        if domain_update:
            domain_updates.append(domain_update)

    for path in domain_randomization_update_paths or []:
        domain_updates.append(dict(_load_json_mapping(path)))

    return merge_pool_dynamics_cfg_updates(
        base_profile,
        cfg_updates=cfg_updates,
        domain_randomization_updates=domain_updates,
        name=name,
        description=description,
        strict=strict,
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must contain a JSON object.")
    for key in data:
        if not isinstance(key, str):
            raise ValueError(f"{path} must contain string keys.")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        profile = build_profile_from_files(
            base_profile_path=args.base_profile,
            update_paths=args.updates,
            domain_randomization_update_paths=args.domain_randomization_updates,
            name=args.name,
            description=args.description,
            strict=not bool(args.allow_unknown),
        )
        write_pool_dynamics_profile_json(profile, args.output)
    except Exception as exc:
        print(f"Failed to build pool profile: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
