# Methods

[Project overview](../README.md) · [Reproduction](REPRODUCTION.md) · [Results](RESULTS.md)

## Public input and physical grid

The ten M1 clips and frequency-domain kernels come from OpenILT at commit `dabb97c6ca3dfd159362e48273c436444c77353b`. Every run records SHA-256 hashes of the original GLP and kernel files. The implementation rasterizes the original polygons at 2048 × 2048 using OpenILT's inclusive OpenCV polygon endpoints, then point-samples `full[::4, ::4]` for a 512-grid optimization. The physical tile remains 2048 nm across; one coarse-grid error pixel represents 16 nm².

This sampling preserves the physical Fourier-frequency indexing, but binary polygon boundaries can change under discretization. The smooth band-limited resampling test checks the frequency convention; it does not establish that binary edge aliasing vanishes.

## Optical model and objectives

For focus or defocus mode $c$, the simulator sums the weighted intensities from all 24 coherent modes:

$$
I_c(M) = \sum_{k=1}^{24} w_{c,k}\left\lvert \mathcal{F}^{-1}\left(H_{c,k}\mathcal{F}(M)\right)\right\rvert^2.
$$

The 35 × 35 kernel centers represent zero frequency. The implementation embeds frequency indices −17 through 17 into matching FFT bins on the current grid. Both transforms use PyTorch's `norm="forward"`: the forward transform divides by the pixel count, while the inverse adds no further normalization.

The three conditions are focus/amplitude 1.00, focus/amplitude 1.02, and defocus/amplitude 0.98. Following the supplied OpenILT code, the amplitude multiplier $d_c$ produces intensity $d_c^2 I_c$. It is not a direct statement about a percentage change in physical exposure energy.

The smooth resist surrogate is

$$
R_c(M) = \mathrm{sigmoid}\left(50\left[d_c^2 I_c(M)-0.225\right]\right).
$$

The nominal objective uses only the nominal condition; the robust objective averages all three. Both minimize the summed squared difference from binary target $T$, normalized by target foreground count. Adam uses learning rate 0.25 for 100 updates. Pixel logits start at +2 inside the target and −2 outside, and a sigmoid maps them to continuous mask values. Only the central 1024 × 1024 nm region may transmit; the surrounding mask is fixed at zero. No total-variation or curvature penalty is used.

## Candidate pool and repair

The binary pool contains steps 0, 5, 10, …, 100: 21 candidates thresholded at 0.5. All raw candidates are scored using three-corner hard print error, even when their training objective is nominal-only. The raw selector minimizes mean hard error, with the earliest checkpoint breaking ties.

The geometric structure is a square with center radius 8 nm. Its coarse-grid radius is 2 pixels, giving a 5 × 5 structure. At 2048 resolution it is 17 × 17, with radius 8 pixels. Both have 16 nm center-to-center span; the discrete structure width is not a process minimum-linewidth claim.

Two checks are independent:

- **Opening defect pixels:** foreground removed by square opening.
- **Closing defect pixels:** background added by square closing.

A mask passes only if both counts are zero. Defects count pixels, rather than connected violations or exposure shots. Exterior pixels are zero; closing explicitly pads an intermediate halo before cropping back to the tile.

Repair repeatedly applies closing after opening, checking both rules after every round, for at most 16 rounds. It detects cycles and composite fixed points. A fixed point of the combined operation does not automatically pass both independent tests. The implementation reports an unresolved status rather than substituting an empty mask.

| Route | Candidate and scoring rule |
|---|---|
| `target` | Use target geometry directly as the mask. |
| `nominal` | Select by raw three-corner hard error along the nominal trajectory. |
| `robust` | Select by raw three-corner hard error along the robust trajectory. |
| `robust_postrepair` | Repair the same selected raw robust candidate. |
| `repair_aware` | Evaluate repaired robust candidates; prefer those passing both rules, then minimize repaired hard error and checkpoint step. If no candidate passes, minimize residual defects, error, then step. |

The three robust variants share the same gradient trajectory and candidate pool. They do not have equal wall time: repair-aware selection incurs repair and extra forward-scoring work. Instrumentation records optimization, raw scoring, repair, and repaired scoring separately.

## Independent evaluation

The final evaluator uses SciPy complex128 transforms and one kernel at a time. It multiplies a centered frequency window after `fftshift`, then applies `ifftshift` before the inverse. It neither calls the PyTorch simulator nor reuses its frequency-scatter arrays.

Hard printing thresholds intensity at 0.225. For condition $c$, the reported pixel count is the XOR between the printed binary set and target. Multiplying by pixel area gives nm². Mean and worst corner errors summarize the three conditions. Process-variation band is the union of all three printed sets minus their intersection; the implementation does not assume high and low conditions are spatially nested.

Boundary distance is a one-way Euclidean distance from target boundary pixels to nominal print boundary pixels. Mean and 95th-percentile values are descriptive proxies. They are not official ICCAD gauge EPE.

## Numerical checks and limits

The checks cover SciPy/PyTorch agreement, zero and constant masks, periodic translation, amplitude-squared scaling, directional finite differences, a smooth fixed-tile resampling fixture, comparison with the upstream exact forward implementation, and a geometry fixture containing both types of defect. A separate audit checks selected masks' corner counts across float32 selection and float64 evaluation. Agreement for the selected masks does not prove that every unselected candidate would have the same double-precision rank.

The model is a public SOCS approximation with an empirical resist surrogate. No full Maxwell solver, resist chemistry calibration, wafer validation, or manufacturing-rule certification is included.
