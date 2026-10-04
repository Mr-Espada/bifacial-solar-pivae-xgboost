"""Install the declared Stan toolchain explicitly, never while importing models."""

import argparse
from pathlib import Path
import cmdstanpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".cmdstan")
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()
    target = Path(args.dir).resolve()
    if not cmdstanpy.install_cmdstan(version="2.37.0", dir=str(target), cores=args.cores):
        raise SystemExit("CmdStan installation failed")
    print(f"Installed CmdStan 2.37.0 under {target}")
