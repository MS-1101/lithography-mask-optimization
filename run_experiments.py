"""Run real public ICCAD13 clips. Report CPU simulation outcomes with explicit protocol and data provenance."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from litho_core import (ROOT, TILE_NM, SOCS, load_target, repair, geometry, evaluate,
                        environment, provenance)


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def optimize(target_np, sim, cfg, robust):
    target = torch.as_tensor(target_np, dtype=sim.dtype, device=sim.device_name)
    logits = torch.nn.Parameter((target*2-1)*cfg["initial_logit"])
    roi = torch.zeros_like(target)
    n = target.shape[0]
    roi[n//4:3*n//4, n//4:3*n//4] = 1
    opt = torch.optim.Adam([logits], lr=cfg["learning_rate"])
    history, checkpoints = [], []
    timings = {"optimization_s": 0., "raw_checkpoint_scoring_s": 0.,
               "repair_s": 0., "repaired_checkpoint_scoring_s": 0.}
    target_mass = target.sum().clamp_min(1)
    for step in range(cfg["steps"]+1):
        if step % cfg["checkpoint_every"] == 0 or step == cfg["steps"]:
            b = ((torch.sigmoid(logits.detach()) * roi) >= .5).cpu().numpy()
            t = time.perf_counter()
            raw = sim.hard_scores(torch.as_tensor(b, dtype=sim.dtype, device=sim.device_name), target)
            timings["raw_checkpoint_scoring_s"] += time.perf_counter()-t
            t = time.perf_counter()
            fixed, repair_stats = repair(b, cfg["radius_px"], cfg["repair_max_rounds"])
            timings["repair_s"] += time.perf_counter()-t
            t = time.perf_counter()
            repaired = sim.hard_scores(torch.as_tensor(fixed, dtype=sim.dtype, device=sim.device_name), target)
            timings["repaired_checkpoint_scoring_s"] += time.perf_counter()-t
            checkpoint = {"step": step, "raw": raw, "repaired": repaired, **repair_stats,
                          "raw_geometry": geometry(b, cfg["radius_px"]), "mask": b,
                          "repaired_mask": fixed}
            checkpoints.append(checkpoint)
            emit("checkpoint", robust=robust, step=step, raw=raw["score_px"],
                 repaired=repaired["score_px"], grid_valid=repair_stats["grid_valid"])
        if step == cfg["steps"]:
            break
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        m = torch.sigmoid(logits)*roi
        resist = sim(m, robust=robust)
        loss = ((resist-target)**2).sum()/len(resist)/target_mass
        loss.backward()
        opt.step()
        elapsed = time.perf_counter()-t
        timings["optimization_s"] += elapsed
        history.append({"step": step+1, "soft_training_loss": loss.item(), "iteration_s": elapsed})
    raw_best = min(checkpoints, key=lambda c: (c["raw"]["score_px"], c["step"]))
    feasible = [c for c in checkpoints if c["grid_valid"]]
    pool = feasible or checkpoints
    repaired_best = min(pool, key=lambda c: (c["geometry_defect_px"],
                                            c["repaired"]["score_px"], c["step"]))
    return {"history": history, "checkpoints": checkpoints, "raw_best": raw_best,
            "repaired_best": repaired_best, "timings": timings}


def save_mask_result(directory, name, mask, target, cfg, metadata):
    metrics, arrays = evaluate(mask, target, cfg["radius_px"], cfg["pixel_nm"], True)
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    Image.fromarray(np.asarray(mask, np.uint8)*255).save(directory/f"{name}_mask.png")
    return {**metadata, "method": name, **metrics}


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pixel = TILE_NM/args.size
    radius = int(round(args.radius_nm/pixel))
    if radius < 1 or abs(radius*pixel-args.radius_nm)>1e-8:
        raise ValueError("radius_nm must be a positive integer number of pixels")
    cfg = {"size": args.size, "physical_tile_nm": TILE_NM, "pixel_nm": pixel,
           "steps": args.steps, "learning_rate": args.lr, "initial_logit": 2.,
           "checkpoint_every": args.checkpoint_every, "radius_nm": args.radius_nm,
           "radius_px": radius, "structure_width_px": 2*radius+1,
           "structure_center_span_nm": 2*radius*pixel, "repair_max_rounds": 16,
           "seed": args.seed, "device": "cpu", "kernel_count": 24,
           "rasterization": "OpenILT full 2048 inclusive fillPoly; point sample full[::stride,::stride]",
           "loss": "mean over 3 corners of squared sigmoid-resist error / target foreground pixels",
           "selection_loss": "mean over 3 corners of hard wafer XOR target pixel count",
           "candidate_tie_break": "earliest checkpoint; repaired selector prefers valid candidates",
           "design_split": {"development": [1,2,3], "held_out": [4,5,6,7,8,9,10]},
           "result_status": "archived CPU simulation experiment",
           "compute_fairness": "robust variants share same gradient trajectory and candidate pool; repair/forward overhead reported separately; not equal wall time"}
    cfg_hash = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    manifest = {"config": cfg, "config_sha256": cfg_hash, "environment": environment(),
                "provenance": provenance(), "clips": args.clips,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    if args.freeze:
        freeze = Path(args.freeze)
        if freeze.exists():
            frozen = json.loads(freeze.read_text())
            if frozen["config_sha256"] != cfg_hash:
                raise ValueError("Frozen protocol mismatch; do not retune on held-out clips")
        else:
            if any(c>3 for c in args.clips):
                raise ValueError("Freeze the protocol on development clips before held-out execution")
            freeze.write_text(json.dumps({"config":cfg,"config_sha256":cfg_hash,
                                         "frozen_utc":manifest["started_utc"]},indent=2,ensure_ascii=False))
    sim = SOCS(args.size)
    rows = []
    for clip in args.clips:
        t_clip = time.perf_counter()
        directory = out / f"M1_test{clip}"
        directory.mkdir(exist_ok=True)
        target = load_target(clip,args.size)
        metadata = {"clip":clip,"split":"development" if clip<=3 else "held_out",
                    "size":args.size,"pixel_nm":pixel,"physical_tile_nm":TILE_NM,
                    "config_sha256":cfg_hash, "device":"cpu"}
        emit("clip_start",clip=clip,foreground_pixels=int(target.sum()),size=args.size)
        rows.append(save_mask_result(directory,"target",target,target,cfg,
                                     {**metadata,"selected_step":0,"optimization_s":0.,
                                      "raw_checkpoint_scoring_s":0.,"repair_s":0.,
                                      "repaired_checkpoint_scoring_s":0.}))
        histories = {}
        for mode in ("nominal","robust"):
            result = optimize(target,sim,cfg,robust=mode=="robust")
            raw, aware = result["raw_best"],result["repaired_best"]
            cp_serial = [{k:v for k,v in c.items() if k not in ("mask","repaired_mask")}
                         for c in result["checkpoints"]]
            histories[mode]={"training":result["history"],"checkpoints":cp_serial,
                             "timings":result["timings"]}
            choices = [(mode,raw["mask"],raw["step"])]
            if mode=="robust":
                choices += [("robust_postrepair",raw["repaired_mask"],raw["step"]),
                            ("repair_aware",aware["repaired_mask"],aware["step"])]
            for name,mask,step in choices:
                row=save_mask_result(directory,name,mask,target,cfg,
                                     {**metadata,"selected_step":step,**result["timings"]})
                rows.append(row)
                emit("result",**row)
            np.savez_compressed(directory/f"{mode}_checkpoints.npz",
                                steps=np.array([c["step"] for c in result["checkpoints"]]),
                                masks=np.stack([c["mask"] for c in result["checkpoints"]]),
                                repaired_masks=np.stack([c["repaired_mask"] for c in result["checkpoints"]]))
        histories["clip_total_s"]=time.perf_counter()-t_clip
        (directory/"trajectory.json").write_text(json.dumps(histories,indent=2))
        (out/"results.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False))
        with (out/"results.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        emit("clip_complete",clip=clip,elapsed_s=histories["clip_total_s"])
    emit("complete",output=str(out),rows=len(rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips",type=lambda s:list(map(int,s.split(","))),default=[1,2,3])
    p.add_argument("--size",type=int,default=512)
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--lr",type=float,default=.25)
    p.add_argument("--checkpoint-every",type=int,default=5)
    p.add_argument("--radius-nm",type=float,default=8.)
    p.add_argument("--seed",type=int,default=20260908)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--out",default=str(ROOT/"results"/"development"))
    p.add_argument("--freeze",default=None)
    run(p.parse_args())
