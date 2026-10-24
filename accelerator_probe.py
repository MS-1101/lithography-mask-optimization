"""Optional actual accelerator forward/gradient parity probe; no speedup claim.

The main statistical experiment is CPU-only. This checks whether the same SOCS
module can execute natively on an available CUDA/MPS backend. Unsupported
operations are recorded explicitly; CPU fallback is not labelled GPU execution.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from litho_core import ROOT,SOCS,load_target,environment


def sync(device):
    if device=="cuda":torch.cuda.synchronize()
    if device=="mps":torch.mps.synchronize()


def run(args):
    torch.set_num_threads(4)
    data={"environment":environment(),"device":args.device,"size":args.size,
          "scope":"functional accelerator parity test, not a speedup benchmark"}
    target=torch.from_numpy(load_target(1,args.size))
    cpu=SOCS(args.size)
    a=(target*.8+.1).requires_grad_(True)
    output=cpu(a);((output-target)**2).mean().backward()
    try:
        acc=SOCS(args.size,device=args.device)
        b=a.detach().to(args.device).requires_grad_(True)
        t=target.to(args.device)
        sync(args.device);started=time.perf_counter()
        output_acc=acc(b);((output_acc-t)**2).mean().backward()
        sync(args.device)
        data.update({"executed":True,"forward_backward_s":time.perf_counter()-started,
                     "forward_max_abs":float((output_acc.detach().cpu()-output.detach()).abs().max()),
                     "gradient_max_abs":float((b.grad.cpu()-a.grad).abs().max()),
                     "gradient_relative_l2":float((b.grad.cpu()-a.grad).norm()/a.grad.norm())})
        data["passed"]=data["forward_max_abs"]<2e-4 and data["gradient_relative_l2"]<2e-3
    except Exception as exc:
        data.update({"executed":False,"passed":False,"error":str(exc)[:1500]})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=["mps","cuda"],default="mps")
    p.add_argument("--size",type=int,default=128)
    p.add_argument("--out",type=Path,default=ROOT/"results"/"accelerator_probe.json")
    run(p.parse_args())
