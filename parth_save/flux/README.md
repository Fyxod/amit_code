# Curated FLUX.2 Klein 4B examples

This folder contains ten visually reviewed paired edits from
`black-forest-labs/FLUX.2-klein-4B`. Every case contains the original input,
saved geometry-perturbed input, clean edit, perturbed edit, and a four-panel
comparison strip. Clean and perturbed edits use identical prompt, seed,
inference-step, and guidance settings.

`metrics.json` records paired SSIM, PSNR, MSE, and L2 values, exact ArcFace
iResNet-100 cosine similarities, source paths and hashes, the archived
perturbation-generation command, and the visual-review observation. The exact
perturbed input is retained in each folder, so the paired editor result can be
replayed directly even if the historical geometry generator changes.

Run `python verify_flux_parth_save.py --root .` to reopen every artifact,
recompute paired image metrics, validate hashes and reproduction settings, and
confirm that all four ArcFace comparisons are present. The resulting audit is
saved as `verification.json`.
