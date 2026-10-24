"""Auditable SOCS lithography on the public OpenILT/ICCAD13 kernels.

This is an independently written implementation. The downloaded data and upstream
reference implementation retain their MIT license. No foundry model is claimed.
"""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy import fft, ndimage
import torch

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "OpenILT"
COMMIT = "dabb97c6ca3dfd159362e48273c436444c77353b"
TILE_NM = 2048
THRESHOLD = 0.225
STEEPNESS = 50.0
CORNERS = (("nominal", "focus", 1.0), ("high", "focus", 1.02),
           ("low", "defocus", 0.98))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    files = sorted((VENDOR / "kernel" / "kernels").glob("*.pt"))
    files += sorted((VENDOR / "kernel" / "scales").glob("*.pt"))
    files += sorted((VENDOR / "benchmark" / "ICCAD2013").glob("*.glp"))
    return {"repository": "https://github.com/OpenOPC/OpenILT", "commit": COMMIT,
            "files": {str(p.relative_to(VENDOR)): sha256(p) for p in files}}


def load_target(clip: int, size: int = 512):
    """Rasterize exactly with upstream GLP/OpenCV at 2048, then point-sample.

    Full-size inclusive polygon endpoint semantics are inherited from OpenILT.
    No per-layout crop, resizing of the physical tile, or synthetic replacement.
    For size 512: T_small[i,j] = T_full[4*i,4*j]. This preserves the full tile's
    Fourier frequencies; edge discretization error remains and is measured.
    """
    if TILE_NM % size or size < 64:
        raise ValueError("size must divide 2048 and be >= 64")
    sys.path.insert(0, str(VENDOR)) if str(VENDOR) not in sys.path else None
    from pycommon.glp import Design
    path = VENDOR / "benchmark" / "ICCAD2013" / f"M1_test{clip}.glp"
    design = Design(str(path), down=1)
    design.center(2048, 2048, 512, 512)
    full = (design.mat(2048, 2048, 512, 512) > .5).astype(np.float32)
    stride = TILE_NM // size
    return full[::stride, ::stride].copy()


def load_kernels(dtype=np.complex128):
    result = {}
    for mode in ("focus", "defocus"):
        k = torch.load(VENDOR / "kernel" / "kernels" / f"{mode}.pt",
                       map_location="cpu", weights_only=True).numpy()
        w = torch.load(VENDOR / "kernel" / "scales" / f"{mode}.pt",
                       map_location="cpu", weights_only=True).numpy()
        result[mode] = (np.moveaxis(k, 2, 0).astype(dtype), w.astype(np.float64))
    return result


class SOCS(torch.nn.Module):
    """Forward FFT uses mean-normalized coefficients; inverse uses no 1/N².

    Kernel array center corresponds to zero spatial frequency. We keep all 24
    modes, not the upstream combined-kernel backward approximation. Focus and
    defocus aerial images are computed once; dose is an amplitude multiplier,
    hence intensity scales as dose², exactly as OpenILT's supplied convention.
    """
    def __init__(self, size, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.size, self.dtype, self.device_name = size, dtype, str(device)
        ctype = torch.complex128 if dtype == torch.float64 else torch.complex64
        kernels = load_kernels()
        for mode, (k, w) in kernels.items():
            # Scatter centered 35×35 modes to the matching FFT frequency bins.
            h = torch.zeros((24, size, size), dtype=ctype, device=device)
            freq = torch.arange(-17, 18, device=device) % size
            h[:, freq[:, None], freq[None, :]] = torch.as_tensor(k, dtype=ctype, device=device)
            self.register_buffer("h_" + mode, h)
            self.register_buffer("w_" + mode, torch.as_tensor(w, dtype=dtype, device=device))

    def aerial_base(self, mask, modes=("focus", "defocus")):
        spectrum = torch.fft.fft2(mask, norm="forward")
        result = {}
        for mode in modes:
            fields = torch.fft.ifft2(spectrum[None] * getattr(self, "h_" + mode),
                                     norm="forward")
            result[mode] = (getattr(self, "w_" + mode)[:, None, None] *
                            (fields.real.square() + fields.imag.square())).sum(0)
        return result

    def forward(self, mask, robust=True):
        base = self.aerial_base(mask, ("focus", "defocus") if robust else ("focus",))
        conditions = CORNERS if robust else CORNERS[:1]
        return torch.stack([torch.sigmoid(STEEPNESS * (base[mode] * dose**2 - THRESHOLD))
                            for _, mode, dose in conditions])

    @torch.no_grad()
    def hard_scores(self, mask, target):
        base = self.aerial_base(mask)
        errors = []
        for _, mode, dose in CORNERS:
            wafer = base[mode] * dose**2 >= THRESHOLD
            errors.append(torch.count_nonzero(wafer != (target > .5)).item())
        return {"corner_error_px": errors, "score_px": sum(errors) / 3.0,
                "worst_error_px": max(errors)}


def reference_aerial(mask, kernels=None):
    """Independent SciPy evaluator, one kernel at a time, complex128.

    It deliberately does not call the torch simulator or reuse its embedding:
    transform+fftshift, multiply a centered 35×35 window, ifftshift, inverse FFT.
    """
    kernels = load_kernels() if kernels is None else kernels
    mask = np.asarray(mask, dtype=np.float64)
    n = mask.shape[0]
    centered = fft.fftshift(fft.fft2(mask, norm="forward"))
    lo, hi = n // 2 - 17, n // 2 + 18
    output = {}
    for mode, (ks, ws) in kernels.items():
        intensity = np.zeros_like(mask)
        for k, w in zip(ks, ws):
            field_f = np.zeros((n, n), dtype=np.complex128)
            field_f[lo:hi, lo:hi] = centered[lo:hi, lo:hi] * k
            field = fft.ifft2(fft.ifftshift(field_f), norm="forward")
            intensity += w * (field.real**2 + field.imag**2)
        output[mode] = intensity
    return output


def _opening(mask, radius):
    return ndimage.binary_opening(mask, structure=np.ones((2*radius+1,)*2, bool),
                                  border_value=0)


def _closing(mask, radius):
    # Zero exterior; pad before closing to avoid eroding the original boundary
    # merely because a finite-array intermediate lacks its exterior halo.
    padded = np.pad(mask, radius * 2, constant_values=False)
    out = ndimage.binary_closing(padded, structure=np.ones((2*radius+1,)*2, bool),
                                 border_value=0)
    r = radius * 2
    return out[r:-r, r:-r] if r else out


def geometry(mask, radius=2):
    b = np.asarray(mask, dtype=bool)
    on = b & ~_opening(b, radius)
    off = _closing(b, radius) & ~b
    a, z = int(on.sum()), int(off.sum())
    return {"opening_defect_px": a, "closing_defect_px": z,
            "geometry_defect_px": a + z, "grid_valid": a == 0 and z == 0}


def repair(mask, radius=2, max_rounds=16):
    """Alternate closing(opening(B)), but check BOTH rules independently.

    A fixed point of their composition is not assumed to satisfy both rules.
    Detect that failure and return status, without blank-mask fallback.
    """
    b = np.asarray(mask, dtype=bool).copy()
    original = b.copy()
    seen = set()
    status = "max_rounds"
    for iteration in range(max_rounds + 1):
        g = geometry(b, radius)
        if g["grid_valid"]:
            status = "valid"
            break
        key = np.packbits(b).tobytes()
        if key in seen:
            status = "cycle_or_composite_fixed_point"
            break
        seen.add(key)
        if iteration < max_rounds:
            b = _closing(_opening(b, radius), radius)
    return b, {**g, "repair_status": status, "repair_rounds": iteration,
               "repair_changed_px": int(np.count_nonzero(original != b))}


def evaluate(mask, target, radius, pixel_nm, save_arrays=False):
    started = time.perf_counter()
    mask, target = np.asarray(mask, bool), np.asarray(target, bool)
    aerial = reference_aerial(mask)
    wafers = np.stack([aerial[mode] * dose**2 >= THRESHOLD for _, mode, dose in CORNERS])
    errors = np.count_nonzero(wafers != target[None], axis=(1, 2))
    pvb = np.any(wafers, axis=0) != np.all(wafers, axis=0)
    # Boundary-distance proxy is not ICCAD's gauge EPE implementation.
    target_edge = target ^ ndimage.binary_erosion(target)
    wafer_edge = wafers[0] ^ ndimage.binary_erosion(wafers[0])
    if target_edge.any() and wafer_edge.any():
        distances = ndimage.distance_transform_edt(~wafer_edge)[target_edge] * pixel_nm
        boundary_mean, boundary_p95 = float(distances.mean()), float(np.quantile(distances, .95))
    else:
        boundary_mean = boundary_p95 = None
    result = {"nominal_error_px": int(errors[0]), "high_error_px": int(errors[1]),
              "low_error_px": int(errors[2]), "mean_corner_error_px": float(errors.mean()),
              "worst_corner_error_px": int(errors.max()),
              "nominal_error_nm2": float(errors[0]*pixel_nm**2),
              "mean_corner_error_nm2": float(errors.mean()*pixel_nm**2),
              "worst_corner_error_nm2": float(errors.max()*pixel_nm**2),
              "pv_union_band_px": int(pvb.sum()), "pv_union_band_nm2": float(pvb.sum()*pixel_nm**2),
              "target_to_nominal_boundary_mean_nm": boundary_mean,
              "target_to_nominal_boundary_p95_nm": boundary_p95,
              "mask_on_px": int(mask.sum()), **geometry(mask, radius),
              "independent_eval_s": time.perf_counter()-started}
    arrays = {"mask": mask, "target": target, "wafer_nominal": wafers[0],
              "wafer_high": wafers[1], "wafer_low": wafers[2],
              "pv_union_band": pvb, "aerial_nominal": aerial["focus"].astype(np.float32)}
    return (result, arrays) if save_arrays else result


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "torch": torch.__version__,
            "numpy": np.__version__, "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads()}
