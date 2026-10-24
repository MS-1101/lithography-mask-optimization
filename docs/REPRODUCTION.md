# Reproduction

[Project overview](../README.md) · [Methods](METHODS.md) · [Results](RESULTS.md)

## Environment and upstream inputs

The archived runs used Python 3.12.14 on macOS 14.6.1 arm64, PyTorch 2.14.0, NumPy 2.5.3, SciPy 1.18.1, OpenCV 5.0.0.93, and Pillow 12.3.0. Core requirements are pinned in [requirements.txt](../requirements.txt); figure regeneration additionally uses Matplotlib 3.11.1. The recorded four-thread CPU timings came from a shared desktop environment and are not controlled throughput measurements.

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
python setup_openilt.py
```

The setup script uses Git to fetch the exact OpenILT commit `dabb97c6ca3dfd159362e48273c436444c77353b`. A sparse checkout retains `benchmark/ICCAD2013`, `kernel`, `pycommon`, and `pylitho`. It checks the 21 layout/kernel hashes recorded in the development manifest and refuses to overwrite a different existing checkout. Network access is needed for this initial fetch. Third-party license text is in [OPENILT_LICENSE](../OPENILT_LICENSE).

## Fast checks

```bash
python test_physics.py --out results/smoke/physics_checks.json
python audit_selection.py --out results/smoke/selection_audit.json
python make_figures.py --out results/regenerated-figures
```

The physics command returns a nonzero status on failed assertions. The audit reads archived checkpoint scores and final metrics; it does not optimize or change selected masks. Figure generation reads recorded masks, trajectories, and metrics, so it needs no training or optical evaluation. The audit and figures can run before downloading OpenILT; the physics checks require it.

To run a small end-to-end optimization smoke check using a new output directory:

```bash
python run_experiments.py --clips 1 --size 64 --steps 2 \
  --checkpoint-every 1 --radius-nm 32 --threads 2 --out results/smoke/run64
```

This command exercises the pipeline at reduced resolution. Its coarse geometry, step count, and results differ from the recorded 512-grid protocol; it is not evidence for the table on the project homepage.

## Reproduce the full experiment

```bash
python reproduce_all.py --out results/reproduction
```

Choose a nonexistent output directory. The wrapper refuses an existing directory and stops at the first failing command. It runs the physics checks, freezes a new 512-grid protocol on development clips, evaluates held-out clips, repeats the two native-grid checks, audits selected scores, and generates figures. It keeps logs under the new directory and preserves the archived evidence.

The optimization commands are equivalent to:

```bash
python run_experiments.py --clips 1,2,3 --size 512 --steps 100 \
  --out results/reproduction/development --freeze results/reproduction/protocol512_main.json
python run_experiments.py --clips 4,5,6,7,8,9,10 --size 512 --steps 100 \
  --out results/reproduction/heldout --freeze results/reproduction/protocol512_main.json
```

Defaults fix learning rate 0.25, checkpoint interval 5, radius 8 nm, seed 20260908, and four CPU threads. The frozen configuration hash must match; a held-out run cannot create its own initial freeze. Use a new protocol and clearly separate results when changing settings.

The repository-level [protocol512_main.json](../protocol512_main.json) contains the archived settings. Descriptive status metadata was normalized for the public archive and the configuration hash recomputed consistently in the protocol, manifests, and metric rows. Computational settings and recorded numerical outcomes were retained.

Individual experiment commands overwrite their specified output directory. Use fresh paths to preserve previous runs.

## Native-grid re-evaluation

The compact archive retains all coarse selected masks, so the two recorded 2048-grid checks can be repeated directly:

```bash
python native_recheck.py --source results/development --clip 1 \
  --out results/reproduction-native1
python native_recheck.py --source results/heldout --clip 4 \
  --out results/reproduction-native4
```

`results/development` and `results/heldout` identify the compact archived run names even though their stored evidence is flat. The reader also accepts the ordinary nested directories produced by a fresh experiment.

Each coarse mask is repeated four times per axis. The target is freshly rasterized from the original GLP at 2048 resolution. All 24 optical modes and the same three corners are used. This is native-grid **evaluation of lifted masks**, rather than native-grid optimization. These commands generate much larger arrays than the compact published archive.

## Artifact layout

The public archive keeps the evidence needed to inspect all 50 coarse-grid outcomes and 420 raw checkpoint candidates without storing derived continuous aerial images.

| Artifact | Contents |
|---|---|
| `results/development_results.{csv,json}` | 15 outcomes: 3 clips × 5 routes. |
| `results/heldout_results.{csv,json}` | 35 outcomes: 7 clips × 5 routes. |
| `results/*_manifest.json` | Configuration, runtime environment, original input hashes, and scope. |
| `results/selected_masks.npz` | 50 selected outcomes, each containing mask, target, three hard wafer maps, and process-variation band. |
| `results/checkpoints.npz` | Raw and repaired masks for 21 checkpoints of each nominal/robust trajectory, plus checkpoint steps. |
| `results/trajectories.json` | Training history, selection scores, geometry checks, and timing ledger for every clip. |
| `results/native_M1_test{1,4}_results.{csv,json}` | Two native-grid evaluations, five routes each. |
| `results/physics_checks.json` | Recorded simulator and derivative checks. |
| `results/selection_audit.json` | Selected-score agreement for 40 optimized outcomes and 120 corner counts. |
| `results/metrics_summary.json` | Held-out summary and paired comparisons. |
| `figures/` | Eight evidence-derived figures and a paired CSV. |

NPZ keys use `run__M1_testN__method__array`, for example `heldout__M1_test6__repair_aware__mask`. Checkpoint keys use `run__M1_testN__robust_checkpoints__masks` or `__repaired_masks`; `__steps` records the corresponding checkpoint numbers. Arrays contain numerical or Boolean values and can be loaded with `allow_pickle=False`.

```python
import numpy as np
from evidence import load_arrays, read_trajectory

arrays = load_arrays("results", "heldout", 6, "repair_aware")
print(arrays["mask"].shape)  # (512, 512)
trajectory = read_trajectory("results", "heldout", 6)
print(trajectory["robust"]["checkpoints"][0]["step"])  # 0
with np.load("results/checkpoints.npz", allow_pickle=False) as archive:
    print(archive["heldout__M1_test6__robust_checkpoints__masks"].shape)
```

The compact arrays retain their source values. Only the redundant `aerial_nominal` arrays are omitted from coarse selected outcomes; all native-grid dense arrays are omitted. Re-evaluation regenerates continuous intensities from masks and verified upstream inputs. The earlier 40-step development pilot is omitted in favor of the complete 100-step experiment. Archived aggregate results remain accessible in plain CSV/JSON.

## Optional accelerator probe

```bash
python accelerator_probe.py --device mps --size 128
```

Use `--device cuda` on a supported NVIDIA environment. The probe compares CPU and accelerator forward/backward functionality and explicitly records unsupported operations. Its single timing observation is not a GPU speedup benchmark. The reported main experiments remain CPU runs.
