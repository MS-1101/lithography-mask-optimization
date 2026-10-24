"""Audit float32 selection scores against independent float64 final evaluation.

This reads existing evidence only; it does not rerun or change selected masks.
"""
import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
from evidence import read_results, read_trajectory


def run(args):
    records=[]
    for run_dir in args.runs:
        run_dir=Path(run_dir)
        rows=read_results(run_dir.parent, run_dir.name)
        for row in rows:
            if row["method"]=="target":continue
            traj=read_trajectory(run_dir.parent, run_dir.name, row["clip"])
            mode="nominal" if row["method"]=="nominal" else "robust"
            cp=next(c for c in traj[mode]["checkpoints"] if c["step"]==row["selected_step"])
            field="repaired" if row["method"] in ("repair_aware","robust_postrepair") else "raw"
            selection=cp[field]["corner_error_px"]
            evaluation=[row[f"{name}_error_px"] for name in ("nominal","high","low")]
            records.append({"run":run_dir.name,"clip":row["clip"],"method":row["method"],
                            "selected_step":row["selected_step"],"torch_float32_error_px":selection,
                            "scipy_float64_error_px":evaluation,
                            "evaluation_minus_selection_px":[b-a for a,b in zip(selection,evaluation)],
                            "exact_match":selection==evaluation})
    data={"scope":"matching selected masks only; not a proof that all candidates have identical float32/float64 rank",
          "records":records,"checked_selected_masks":len(records),
          "exactly_matching_masks":sum(r["exact_match"] for r in records),
          "mismatching_masks":sum(not r["exact_match"] for r in records),
          "max_absolute_corner_difference_px":max((abs(d) for r in records for d in r["evaluation_minus_selection_px"]),default=0)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k!="records"},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",nargs="+",default=[str(ROOT/"results"/"development"),str(ROOT/"results"/"heldout")])
    p.add_argument("--out",type=Path,default=ROOT/"results"/"selection_audit.json")
    run(p.parse_args())
