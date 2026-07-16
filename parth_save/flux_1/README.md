# FLUX.2 Klein 4B exploratory examples (flux_1)

This folder preserves the first ten visually reviewed paired edits from
`black-forest-labs/FLUX.2-klein-4B`. Every case contains the original input,
saved geometry-perturbed input, clean edit, perturbed edit, and a four-panel
comparison strip. Clean and perturbed edits use identical prompt, seed,
inference-step, and guidance settings.

These examples are retained as exploratory results, not as strong attack
successes. Several show edit styling or structure changes without an obvious
identity collapse or major whole-image failure. The stricter second search is
stored separately under `parth_save/flux_2` and does not accept minor beard,
accessory, color, pose, styling, or composition variation.

`metrics.json` records paired SSIM, PSNR, MSE, and L2 values, exact ArcFace
iResNet-100 cosine similarities, source paths and hashes, the archived
perturbation-generation command, and the visual-review observation. The exact
perturbed input is retained in each folder, so the paired editor result can be
replayed directly even if the historical geometry generator changes.

Run `python verify_flux_parth_save.py --root . --save-root parth_save/flux_1`
to reopen every artifact,
recompute paired image metrics, validate hashes and reproduction settings, and
confirm that all four ArcFace comparisons are present. The resulting audit is
saved as `verification.json`.
