"""Rebuild every scientific figure from archived measured arrays and JSON.

Run from this directory: python make_figures.py
No generated or illustrative performance data are used.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from evidence import read_results, read_trajectory, load_arrays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
METHODS = ["target", "nominal", "robust", "robust_postrepair", "repair_aware"]
LABELS = ["Target as mask", "Nominal objective", "Robust, raw", "Robust + repair", "Repair-aware"]
SHORT = ["Target", "Nominal", "Robust raw", "Post-repair", "Repair-aware"]
COLORS = ["#718096", "#c98b23", "#3975a4", "#9263ab", "#158477"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 130,
                     "savefig.dpi": 300, "svg.fonttype": "none"})


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def subset(rows, clip, method):
    return next(x for x in rows if x["clip"] == clip and x["method"] == method)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    data_root = args.results_root.resolve()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = read_results(data_root, "development")
    held = read_results(data_root, "heldout")
    assert len(dev) == 15 and len(held) == 35, "Require complete frozen runs"
    clips = sorted({r["clip"] for r in held})
    assert clips == list(range(4, 11))
    hashes = {r["config_sha256"] for r in dev + held}
    assert len(hashes) == 1
    summary = {"status": "archived CPU simulation experiment",
               "development_clips": [1, 2, 3], "held_out_clips": clips,
               "config_sha256": hashes.pop(), "methods": {}}
    for method in METHODS:
        rows = [r for r in held if r["method"] == method]
        summary["methods"][method] = {
            "n": len(rows), "grid_valid_count": sum(r["grid_valid"] for r in rows),
            **{key: float(np.mean([r[key] for r in rows])) for key in
               ["nominal_error_nm2", "mean_corner_error_nm2", "worst_corner_error_nm2",
                "pv_union_band_nm2", "geometry_defect_px", "target_to_nominal_boundary_p95_nm"]}}
    paired = []
    for clip in clips:
        a, b = subset(held, clip, "robust_postrepair"), subset(held, clip, "repair_aware")
        paired.append({"clip": clip, "postrepair_step": a["selected_step"],
                       "aware_step": b["selected_step"],
                       "postrepair_mean_nm2": a["mean_corner_error_nm2"],
                       "aware_mean_nm2": b["mean_corner_error_nm2"],
                       "delta_nm2": b["mean_corner_error_nm2"]-a["mean_corner_error_nm2"],
                       "postrepair_defect_px": a["geometry_defect_px"],
                       "aware_defect_px": b["geometry_defect_px"],
                       "postrepair_valid": a["grid_valid"], "aware_valid": b["grid_valid"]})
    summary["paired"] = paired
    summary["comparison"] = {
        "strict_error_improvement": sum(r["delta_nm2"] < -1e-8 for r in paired),
        "equal_error": sum(abs(r["delta_nm2"]) <= 1e-8 for r in paired),
        "strict_error_increase": sum(r["delta_nm2"] > 1e-8 for r in paired),
        "mean_delta_nm2": float(np.mean([r["delta_nm2"] for r in paired]))}
    timings = [read_trajectory(data_root, "heldout", c)["robust"]["timings"] for c in clips]
    summary["mean_robust_stage_seconds"] = {k: float(np.mean([r[k] for r in timings])) for k in timings[0]}
    summary["mean_full_pipeline_seconds"] = float(np.mean([
        read_trajectory(data_root, "heldout", c)["clip_total_s"] for c in clips]))
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT / "paired_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)

    # Fixed example: first development layout; chosen before held-out results.
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), constrained_layout=True)
    data = {m: load_arrays(data_root, "development", 1, m) for m in METHODS}
    crop = np.s_[128:384, 128:384]
    for ax, m in zip(axes[0], ["target", "robust_postrepair", "repair_aware"]):
        ax.imshow(data[m]["mask"][crop], cmap="gray_r", origin="lower", extent=[512,1536,512,1536], interpolation="nearest")
        ax.set_title({"target":"Target geometry", "robust_postrepair":"Mask: raw-best then repair", "repair_aware":"Mask: feasible candidate selection"}[m])
    for ax, m in zip(axes[1], ["target", "robust_postrepair", "repair_aware"]):
        a = data[m]; t = a["target"].astype(bool); w = a["wafer_nominal"].astype(bool)
        colors = np.zeros(t.shape, dtype=int); colors[t & w] = 1; colors[t & ~w] = 2; colors[~t & w] = 3
        ax.imshow(colors[crop], cmap=ListedColormap(["#ffffff", "#cbd5e0", "#2975b3", "#db704d"]), vmin=0, vmax=3,
                  origin="lower", extent=[512,1536,512,1536], interpolation="nearest")
        ax.set_title("Nominal wafer overlay")
    for ax in axes.flat:
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.suptitle("M1_test1 (development): central 1024 nm view; metrics use the full 2048 nm tile\nGray: matched foreground | Blue: missing print | Orange: excess print", fontsize=12)
    save(fig, "01_mask_and_wafer")

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(clips)); width=.36
    for j,m in enumerate(["robust_postrepair","repair_aware"]):
        values = [subset(held,c,m)["mean_corner_error_nm2"]/1000 for c in clips]
        ax.bar(x+(j-.5)*width, values, width, label=LABELS[METHODS.index(m)], color=COLORS[METHODS.index(m)])
    ax.set(xticks=x, xticklabels=[f"M1_{c}" for c in clips], ylabel="Mean corner XOR area (10³ nm²)",
           title="Held-out layouts: final printed error after geometric repair")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    save(fig, "02_held_out_error")

    tr = read_trajectory(data_root, "development", 1)["robust"]["checkpoints"]
    steps = [r["step"] for r in tr]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(steps, [r["raw"]["score_px"]*16/1000 for r in tr], "o-", ms=3, color=COLORS[2], label="Raw candidate")
    axes[0].plot(steps, [r["repaired"]["score_px"]*16/1000 for r in tr], "s-", ms=3, color=COLORS[3], label="Repaired candidate")
    for m,col in [("robust_postrepair",COLORS[3]),("repair_aware",COLORS[4])]:
        row = subset(dev,1,m); axes[0].scatter(row["selected_step"],row["mean_corner_error_nm2"]/1000,s=125,
            marker="*",color=col,edgecolors="black",zorder=5,label=LABELS[METHODS.index(m)]+" selected")
    axes[0].set(ylabel="Mean corner XOR area (10³ nm²)", title="M1_test1: optical progress does not certify geometric feasibility")
    axes[0].legend(frameon=False, fontsize=8)
    defects=[r["geometry_defect_px"] for r in tr]
    axes[1].plot(steps, defects, "o-", color=COLORS[3],ms=4)
    valid=[r for r in tr if r["grid_valid"]]
    axes[1].scatter([r["step"] for r in valid],[0]*len(valid),color=COLORS[4],s=25,zorder=4,label="Both checks pass")
    axes[1].set(xlabel="Optimization checkpoint (step)",ylabel="Post-repair defect pixels",yscale="symlog")
    axes[1].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.2)
    save(fig,"03_checkpoint_tradeoff")

    fig, ax=plt.subplots(figsize=(9,5.4),constrained_layout=True)
    for m,marker in [("robust_postrepair","s"),("repair_aware","o")]:
        rr=[subset(held,c,m) for c in clips]
        ax.scatter([r["geometry_defect_px"] for r in rr],[r["mean_corner_error_nm2"]/1000 for r in rr],
                   marker=marker,color=COLORS[METHODS.index(m)],s=60,label=LABELS[METHODS.index(m)],alpha=.8)
        for r in rr:
            peer = subset(held,r["clip"],"robust_postrepair")
            if m == "repair_aware" and r["selected_step"] == peer["selected_step"]:
                continue
            offset = (7,-9) if m == "repair_aware" and r["clip"] == 6 else (5,5)
            ax.annotate(str(r["clip"]),(r["geometry_defect_px"],r["mean_corner_error_nm2"]/1000),
                        xytext=offset,textcoords="offset points",fontsize=8)
    ax.set(xscale="symlog",xlabel="Final geometry defect pixels (symmetric-log scale)",ylabel="Mean corner XOR area (10³ nm²)",
           title="Feasibility and fidelity are separate outcomes (labels: held-out clip IDs)")
    ax.legend(frameon=False);ax.grid(alpha=.2)
    save(fig,"04_feasibility_fidelity")

    fig, ax=plt.subplots(figsize=(10,4.8),constrained_layout=True)
    corner_keys=["nominal_error_px","high_error_px","low_error_px"]
    matrix=np.array([[np.mean([subset(held,c,m)[k]*16/1000 for c in clips]) for k in corner_keys] for m in METHODS])
    im=ax.imshow(matrix,cmap="YlGnBu",aspect="auto")
    for i in range(5):
        for j in range(3): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",color="white" if matrix[i,j] > (matrix.min()+matrix.max())/2 else "#172b4d")
    ax.set(xticks=range(3),xticklabels=["Focus / amplitude 1.00","Focus / amplitude 1.02","Defocus / amplitude 0.98"],
           yticks=range(5),yticklabels=LABELS,title="Mean XOR area over seven held-out layouts (10³ nm²)")
    fig.colorbar(im,ax=ax,shrink=.85,label="Area (10³ nm²)")
    save(fig,"05_process_corners")

    fig, ax=plt.subplots(figsize=(10,4.8),constrained_layout=True)
    left=np.zeros(len(clips))
    keys=["optimization_s","raw_checkpoint_scoring_s","repair_s","repaired_checkpoint_scoring_s"]
    names=["Gradient trajectory","Raw scoring","Geometry repair","Repaired scoring"]
    for k,name,col in zip(keys,names,["#3975a4","#91b5cf","#bd96ce","#158477"]):
        values=np.array([r[k] for r in timings]);ax.barh(range(7),values,left=left,label=name,color=col);left+=values
    ax.set(yticks=range(7),yticklabels=[f"M1_{c}" for c in clips],xlabel="Measured CPU wall time (s)",
           title="Robust trajectory instrumentation: selection overhead is included")
    ax.set_ylim(-.6,7.6)
    ax.legend(frameon=False,loc="upper center",ncol=4,fontsize=9)
    fig.supxlabel("Shared trajectory; not an equal-wall-time algorithm benchmark. Excludes final independent evaluation and file I/O.",fontsize=9)
    save(fig,"06_runtime_accounting")

    checks=read(data_root/"physics_checks.json")["checks"]
    grad=checks["directional_gradient"]
    fig,ax=plt.subplots(figsize=(7.5,4.7),constrained_layout=True)
    ax.loglog([r["epsilon"] for r in grad],[r["relative_error"] for r in grad],"o-",color=COLORS[2])
    ax.set(xlabel="Central finite-difference step epsilon",ylabel="Directional derivative relative error",
           title="Gradient verification on a 64 × 64 double-precision fixture")
    ax.grid(which="both",alpha=.2)
    save(fig,"07_gradient_check")

    native=read_results(data_root, "native_M1_test1")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    methods=METHODS[1:]; x=np.arange(4);width=.36
    for i,(rows,pix,name) in enumerate([(dev,4,"512² evaluation"),(native,1,"2048² re-evaluation")]):
        rr=[subset(rows,1,m) for m in methods]
        axes[0].bar(x+(i-.5)*width,[r["mean_corner_error_nm2"]/1000 for r in rr],width,label=name,color=["#3975a4","#c98b23"][i])
        axes[1].bar(x+(i-.5)*width,[r["geometry_defect_px"]*pix**2 for r in rr],width,label=name,color=["#3975a4","#c98b23"][i])
    for ax in axes:
        ax.set_xticks(x,SHORT[1:],rotation=18);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean corner XOR area (10³ nm²)");axes[1].set_ylabel("Geometry defect area (nm²)")
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("M1_test1: native-grid audit of lifted masks; optimization remains at 512²",fontsize=12)
    save(fig,"08_native_grid_audit")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
