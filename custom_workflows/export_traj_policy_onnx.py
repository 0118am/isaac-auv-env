#!/usr/bin/env python3
"""Export a WarpAUV trajectory RSL-RL actor checkpoint to ONNX."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trajectory policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model_*.pt checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("exported_policies"))
    parser.add_argument("--prefix", default="warpauv_traj_policy")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--obs-dim", type=int, default=28)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--activation", default="elu")
    return parser.parse_args()


def build_policy(obs_dim: int, action_dim: int, hidden_dims: list[int], activation: str) -> ActorCritic:
    obs = TensorDict({"policy": torch.zeros(1, obs_dim), "critic": torch.zeros(1, obs_dim)}, batch_size=[1])
    return ActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=action_dim,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation=activation,
        init_noise_std=1.0,
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    policy = build_policy(args.obs_dim, args.action_dim, args.hidden_dims, args.activation)
    policy.load_state_dict(state_dict)
    policy.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    output_path = args.output_dir / f"{args.prefix}_{args.date}_{stem}.onnx"
    dummy_obs = torch.zeros(1, args.obs_dim)
    torch.onnx.export(
        policy.actor,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(output_path)


if __name__ == "__main__":
    main()
