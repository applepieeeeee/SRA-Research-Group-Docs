from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cake_latest.dataset import BoundaryStripDatasetConfig, build_boundary_strip_dataset, save_boundary_dataset
#IF YOU ARE RUNNING THIS LOCALLY PLEASE CHANGE THIS IMPORT COMMAND TO PULL FROM WHEREVER YOU
#SAVED DATASET.PY (IT WILL NOT WORK IF YOU DON'T DO THIS)

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Build a strip-level CAKE boundary-history dataset.")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--partition-rows", type=int, default=2)
    parser.add_argument("--partition-cols", type=int, default=2)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.55])
    parser.add_argument("--num-instances", type=int, default=50)
    parser.add_argument("--burn-in-steps", type=int, default=128)
    parser.add_argument("--trajectory-steps", type=int, default=5000)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--include-remote-interior", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "datasets")
    parser.add_argument("--stem", type=str, default="boundary_strip_dataset")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    config = BoundaryStripDatasetConfig(
        rows=args.rows,
        cols=args.cols,
        partition_rows=args.partition_rows,
        partition_cols=args.partition_cols,
        beta_values=tuple(args.betas),
        num_instances=args.num_instances,
        burn_in_steps=args.burn_in_steps,
        trajectory_steps=args.trajectory_steps,
        history_length=args.history_length,
        horizon=args.horizon,
        window_stride=args.window_stride,
        include_remote_interior=args.include_remote_interior,
        seed=args.seed,
    )

    bundle = build_boundary_strip_dataset(config)
    dataset_path, metadata_path = save_boundary_dataset(bundle, args.output_dir, stem=args.stem)

    summary = {
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
        "train_x_dynamic_shape": list(bundle.arrays["train_x_dynamic"].shape),
        "train_x_static_shape": list(bundle.arrays["train_x_static"].shape),
        "train_y_next_shape": list(bundle.arrays["train_y_next"].shape),
        "train_y_future_shape": list(bundle.arrays["train_y_future"].shape),
        "val_x_dynamic_shape": list(bundle.arrays["val_x_dynamic"].shape),
        "test_x_dynamic_shape": list(bundle.arrays["test_x_dynamic"].shape),
        "strip_width": int(bundle.metadata["strip_width"]),
        "x_static_dim": int(bundle.metadata["x_static_dim"]),
        "num_directed_boundary_strips": int(bundle.metadata["num_directed_boundary_strips"]),
        "raw_windows_per_strip": int(bundle.metadata["raw_windows_per_strip"]),
        "retained_windows_per_strip": int(bundle.metadata["retained_windows_per_strip"]),
        "samples_per_run": int(bundle.metadata["samples_per_run"]),
        "raw_total_samples": int(bundle.metadata["raw_total_samples"]),
        "total_samples": int(bundle.metadata["total_samples"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
