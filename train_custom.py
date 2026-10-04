"""Compatibility entry point. Original experiment is archived under legacy/."""

from pathlib import Path
import sys

if __name__ == "__main__":
    from pivae_hybrid.cli import main
    if len(sys.argv) == 1:
        main(["run-all", "--config", str(Path(__file__).parent / "configs" / "reconstruction.json")])
    else:
        main()
