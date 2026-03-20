from __future__ import annotations

import argparse
import sys

from solarflare_app.modeling.bundle import validate_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="./bundle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_bundle(args.bundle_dir)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(1)
    print("Bundle is valid.")


if __name__ == "__main__":
    main()
