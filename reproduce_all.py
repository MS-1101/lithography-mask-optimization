"""Full CPU reproduction in a new directory, without overwriting archived runs."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results/reproduction")
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"Choose a new output directory; already exists: {out}")
    out.mkdir(parents=True)
    protocol = out / "protocol512_main.json"

    def run(script, *values):
        command = [sys.executable, str(ROOT / script), *map(str, values)]
        print("RUN", command, flush=True)
        with (out / (script.removesuffix(".py") + ".log")).open("a") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    run("test_physics.py", "--out", out / "physics_checks.json")
    for clips, name in [("1,2,3", "development"), ("4,5,6,7,8,9,10", "heldout")]:
        run("run_experiments.py", "--clips", clips, "--size", 512, "--steps", 100,
            "--lr", .25, "--checkpoint-every", 5, "--radius-nm", 8,
            "--seed", 20260908, "--threads", 4, "--out", out / name, "--freeze", protocol)
    run("native_recheck.py", "--source", out / "development", "--clip", 1,
        "--out", out / "native_M1_test1")
    run("native_recheck.py", "--source", out / "heldout", "--clip", 4,
        "--out", out / "native_M1_test4")
    run("audit_selection.py", "--runs", out / "development", out / "heldout",
        "--out", out / "selection_audit.json")
    run("make_figures.py", "--results-root", out, "--out", out / "figures")
    print(f"Completed. Fresh evidence is at {out}. Archived figures were not replaced.")


if __name__ == "__main__":
    main()
