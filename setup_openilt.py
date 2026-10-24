"""Fetch the pinned public OpenILT dependency and verify every recorded input hash."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'vendor' / 'OpenILT'
REPOSITORY = 'https://github.com/OpenOPC/OpenILT.git'
COMMIT = 'dabb97c6ca3dfd159362e48273c436444c77353b'


def git(*args, capture=False):
    result = subprocess.run(['git', '-C', str(DEST), *args], check=True,
                            text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def main():
    if DEST.exists():
        if not (DEST / '.git').exists():
            raise SystemExit('Existing vendor/OpenILT is not a Git checkout; move it before setup.')
        if git('rev-parse', 'HEAD', capture=True) != COMMIT:
            raise SystemExit('Existing OpenILT has a different commit; setup will not overwrite it.')
    else:
        DEST.mkdir(parents=True)
        git('init')
        git('remote', 'add', 'origin', REPOSITORY)
        git('sparse-checkout', 'init', '--cone')
        git('sparse-checkout', 'set', 'benchmark/ICCAD2013', 'kernel', 'pycommon', 'pylitho')
        git('fetch', '--depth=1', 'origin', COMMIT)
        git('checkout', '--detach', 'FETCH_HEAD')
    recorded = json.loads((ROOT / 'results/development_manifest.json').read_text())['provenance']
    if recorded['commit'] != COMMIT:
        raise SystemExit('Recorded input commit does not match setup pin.')
    for name, expected in recorded['files'].items():
        path = DEST / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Input hash mismatch: {name}')
    print(f'Verified OpenILT {COMMIT}: {len(recorded["files"])} recorded inputs.')


if __name__ == '__main__':
    main()
