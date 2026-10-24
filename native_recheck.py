"""2048² independent validation of a 512² optimized mask, with explicit scope.

This does not call the low-resolution run a native-resolution optimization.
Targets are freshly rasterized from the original GLP; masks are 4x nearest
replicated. All 24 supplied optical modes and the same three corners are used.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from litho_core import ROOT,load_target,evaluate,environment,provenance
from evidence import load_arrays


def run(args):
    torch.set_num_threads(args.threads)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    target=load_target(args.clip,2048)
    rows=[]
    manifest={"environment":environment(),"provenance":provenance(),"clip":args.clip,
              "physical_tile_nm":2048,"pixel_nm":1,"evaluation_size":2048,
              "optimization_size":512,"scope":"native-grid re-evaluation; not native-grid optimization",
              "mask_lift":"nearest replication: repeat 4 in each axis",
              "target":"fresh original OpenILT GLP rasterization at 2048",
              "geometry_radius_nm":8,"geometry_radius_px":8,"kernel_count":24,
              "source_run":Path(args.source).name,
              "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    for method in ("target","nominal","robust","robust_postrepair","repair_aware"):
        if method=="target":
            mask=target
        else:
            packed=load_arrays(Path(args.source).parent, Path(args.source).name, args.clip, method)
            small=packed["mask"]
            assert small.shape==(512,512)
            mask=np.repeat(np.repeat(small,4,0),4,1)
        metrics,arrays=evaluate(mask,target,8,1.,True)
        np.savez_compressed(out/f"{method}.npz",**arrays)
        rows.append({"clip":args.clip,"method":method,"size":2048,"pixel_nm":1.,**metrics})
        print(json.dumps(rows[-1]),flush=True)
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    with (out/"results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--clip",type=int,default=1)
    p.add_argument("--out",default=str(ROOT/"results"/"native_M1_test1"))
    p.add_argument("--threads",type=int,default=4)
    run(p.parse_args())
