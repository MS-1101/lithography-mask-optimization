"""Read compact archived evidence or the directory tree produced by a fresh run."""
from pathlib import Path
import json
import numpy as np


def read_results(root, run):
    root = Path(root)
    nested = root / run / 'results.json'
    return json.loads((nested if nested.exists() else root / f'{run}_results.json').read_text())


def read_trajectory(root, run, clip):
    root = Path(root)
    nested = root / run / f'M1_test{clip}' / 'trajectory.json'
    if nested.exists():
        return json.loads(nested.read_text())
    return json.loads((root / 'trajectories.json').read_text())[run][str(clip)]


def load_arrays(root, run, clip, method):
    root = Path(root)
    nested = root / run / f'M1_test{clip}' / f'{method}.npz'
    if nested.exists():
        with np.load(nested, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    prefix = f'{run}__M1_test{clip}__{method}__'
    with np.load(root / 'selected_masks.npz', allow_pickle=False) as data:
        arrays = {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}
    if not arrays:
        raise FileNotFoundError(f'No archived arrays for {run}/M1_test{clip}/{method}')
    return arrays
