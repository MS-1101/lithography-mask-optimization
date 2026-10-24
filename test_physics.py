"""Meaningful simulator, derivative, geometry and provenance checks."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from litho_core import (ROOT,VENDOR,SOCS,CORNERS,THRESHOLD,STEEPNESS,load_kernels,
                        load_target,reference_aerial,geometry,repair,environment,provenance)


def run(out):
    torch.set_num_threads(4)
    rng=np.random.default_rng(173)
    results={"environment":environment(),"provenance":provenance(),"checks":{}}
    checks=results["checks"]
    n=64
    sim=SOCS(n,dtype=torch.float64)
    a=torch.as_tensor(rng.uniform(.05,.95,(n,n)),dtype=torch.float64)
    base=sim.aerial_base(a)
    reference=reference_aerial(a.numpy())
    checks["independent_scipy_forward_max_abs"]={m:float(np.max(np.abs(base[m].detach().numpy()-reference[m])))
                                                for m in reference}
    assert max(checks["independent_scipy_forward_max_abs"].values())<1e-10
    zero=sim.aerial_base(torch.zeros_like(a))
    checks["zero_mask_max_intensity"]=max(x.abs().max().item() for x in zero.values())
    assert checks["zero_mask_max_intensity"]==0
    kernels=load_kernels()
    constant=sim.aerial_base(torch.ones_like(a))
    checks["constant_mask_dc_error"]={m:float((constant[m]-np.sum(w*np.abs(k[:,17,17])**2)).abs().max())
                                       for m,(k,w) in kernels.items()}
    assert max(checks["constant_mask_dc_error"].values())<1e-10
    shifted=sim.aerial_base(torch.roll(a,(3,-7),(0,1)))
    checks["periodic_translation_max_abs"]={m:float((shifted[m]-torch.roll(base[m],(3,-7),(0,1))).abs().max())
                                            for m in base}
    assert max(checks["periodic_translation_max_abs"].values())<1e-10
    dose=1.02
    scaled=sim.aerial_base(a*dose)
    checks["dose_squared_scaling_max_abs"]={m:float((scaled[m]-dose**2*base[m]).abs().max()) for m in base}
    assert max(checks["dose_squared_scaling_max_abs"].values())<1e-10
    # Objective gradient: central finite differences, several step sizes.
    target=torch.as_tensor((rng.random((n,n))>.5).astype(float))
    a.requires_grad_(True)
    f=lambda x:((sim(x)-target)**2).mean()
    value=f(a);value.backward()
    direction=torch.as_tensor(rng.normal(size=(n,n)),dtype=torch.float64)
    direction/=direction.norm()
    analytic=float((a.grad*direction).sum())
    fd=[]
    for eps in (1e-2,1e-3,1e-4,1e-5):
        finite=float((f(a.detach()+eps*direction)-f(a.detach()-eps*direction))/(2*eps))
        fd.append({"epsilon":eps,"finite_difference":finite,"adjoint":analytic,
                   "relative_error":abs(finite-analytic)/max(abs(finite),abs(analytic),1e-12)})
    checks["directional_gradient"]=fd
    assert min(d["relative_error"] for d in fd)<2e-6
    # Same physical Fourier frequencies under a 4x sample-rate change.
    # This is a smooth bandlimited-mask test, NOT a claim that binary edges have no aliasing.
    ns,nl=64,256
    yy,xx=np.meshgrid(np.arange(nl),np.arange(nl),indexing="ij")
    smooth=.35+.15*np.cos(2*np.pi*xx/nl)+.1*np.sin(2*np.pi*yy/nl)
    large=reference_aerial(smooth)
    small=reference_aerial(smooth[::4,::4])
    checks["fixed_tile_bandlimited_resampling_max_abs"]={m:float(np.max(np.abs(large[m][::4,::4]-small[m]))) for m in large}
    assert max(checks["fixed_tile_bandlimited_resampling_max_abs"].values())<1e-10
    # Independent upstream exact forward check; do not use upstream custom backward.
    sys.path.insert(0,str(VENDOR))
    from pylitho.exact import LithoSim
    upstream=LithoSim({"KernelDir":str(VENDOR/"kernel"),"KernelNum":24,
                       "TargetDensity":.225,"PrintThresh":.5,"PrintSteepness":50.,
                       "DoseMax":1.02,"DoseMin":.98,"DoseNom":1.})
    real=load_target(1,256)
    inp=torch.from_numpy(real)
    t=time.perf_counter()
    ours=SOCS(256)(inp).detach()
    theirs=torch.stack(upstream(inp)).detach()
    checks["upstream_exact_forward_max_abs"]=float((ours-theirs).abs().max())
    checks["upstream_forward_comparison_s"]=time.perf_counter()-t
    assert checks["upstream_exact_forward_max_abs"]<2e-5
    # Geometry defects and distinct rules; count pixels, not abstract violations.
    geom=np.zeros((64,64),bool);geom[10:35,10:35]=1;geom[18:20,18:20]=0;geom[45,45]=1
    before=geometry(geom,2);fixed,after=repair(geom,2)
    checks["geometry_fixture"]={"before":before,"after":after,
                                "independent_recheck":geometry(fixed,2)}
    assert before["opening_defect_px"]>0 and before["closing_defect_px"]>0
    assert after["grid_valid"]==geometry(fixed,2)["grid_valid"]
    assert set(np.unique(fixed)).issubset({False,True})
    # Device support is reported, not silently called GPU execution.
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        torch.fft.fft2(torch.ones((16,16),device="mps"))
        checks["mps_complex_fft_probe"]={"supported":True}
    except Exception as exc:
        checks["mps_complex_fft_probe"]={"supported":False,"error":str(exc)[:700]}
    checks["all_10_layouts_present"]=all((VENDOR/"benchmark"/"ICCAD2013"/f"M1_test{i}.glp").exists() for i in range(1,11))
    assert checks["all_10_layouts_present"]
    results["status"]="passed"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2))
    print(json.dumps({"status":"passed","output":str(out),"checks":checks},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT/"results"/"physics_checks.json")
    run(p.parse_args().out)
