"""
Prints the five model checkpoint paths the registry resolves to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify every checkpoint exists on disk; non-zero exit if not")
    ap.add_argument("--docker", action="store_true",
                    help="emit a docker-compose `environment:` block")
    ap.add_argument("--volumes", action="store_true",
                    help="emit docker-compose read-only volume mounts")
    ap.add_argument("--container-root", default="/app/models",
                    help="model root inside the container (default: /app/models)")
    args = ap.parse_args()

    names = registry.model_names()
    missing: list[str] = []

    if args.docker or args.volumes:
        cfg = registry.models_config()["models"]
        if args.docker:
            print(f"      SSC_MODEL_ROOT: {args.container_root}")
            for n in names:
                print(f"      SSC_MODEL_{n.upper()}: {args.container_root}/{cfg[n]['checkpoint']}")
        if args.volumes:
            for n in names:
                rel = cfg[n]["checkpoint"]
                print(f"      - ./models/{rel}:{args.container_root}/{rel}:ro")
        return 0

    for name in names:
        spec = registry.get_model(name)
        exists = spec.checkpoint.exists()
        if not exists:
            missing.append(name)
        line = f"SSC_MODEL_{name.upper()}={spec.checkpoint}"
        if args.check:
            size = f"{spec.checkpoint.stat().st_size / 1e6:.1f} MB" if exists else "MISSING"
            line = f"{line}    # {spec.task:8s} {size}"
        print(line)

    if args.check and missing:
        print(f"\n{len(missing)} checkpoint(s) missing: {', '.join(missing)}",
              file=sys.stderr)
        print("Train them with:  " +
              "  ".join(f"python -m training.train_{n}" for n in missing),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
