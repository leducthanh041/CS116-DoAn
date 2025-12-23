"""build_stage2_feature_label.py

CLI driver to build Stage-2 feature-label parquet.

Usage:
  python build_stage2_feature_label.py --config ./stage2_params.json

Outputs:
  - <out_dir>/feature_label_final.parquet
  - plus intermediate files if enabled.
"""

import argparse
import os

from stage2_feature_label import load_params, build_feature_label_pipeline, log


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="stage2_params.json", help="Path to stage2_params.json")
    return p.parse_args()


def main():
    args = parse_args()
    params = load_params(args.config)

    os.makedirs(params.out_dir, exist_ok=True)

    log("==== Stage2 Feature-Label Builder ====")
    log(f"Config path: {args.config}")
    log(f"Output dir : {params.out_dir}")

    final_path = build_feature_label_pipeline(params)
    log(f"FINAL parquet: {final_path}")


if __name__ == "__main__":
    main()
