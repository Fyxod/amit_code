# Legacy FLUX sweep audit

The existing 300-row FLUX.2 Klein prompt sweep was rescored with the exact
ArcFace iResNet-100 checkpoint and reviewed before starting the new strict
search.

- Rows rescored: 300
- Review gate: input SSIM at least 0.88, original-versus-perturbed ArcFace
  similarity at least 0.80, and original-versus-clean-edit ArcFace similarity
  at least 0.60
- Lowest clean-edit-versus-perturbed-edit ArcFace similarity after that gate:
  approximately 0.696 (`image_1`, `Add a blue baseball cap`, `all_10`)
- Visual verdict: rejected. The edit remains coherent, the requested cap is
  still present, and the face is not visibly collapsed or globally disrupted.

The other high-ranked rows similarly contained accessory, color, smoothing,
or composition differences rather than obvious identity collapse or severe
whole-image failure. None of these legacy rows was promoted into `flux_2`.

Normalized exact metrics and the corresponding review sheet are stored at:

- `targeted_experiments/flux_klein_prompt_sweep/flux_prompt_sweep_arcface.csv`
- `targeted_experiments/flux_klein_prompt_sweep/flux_prompt_sweep_arcface_ranked.csv`
- `targeted_experiments/flux_klein_prompt_sweep/flux_prompt_sweep_arcface_ranked_sheet.jpg`
