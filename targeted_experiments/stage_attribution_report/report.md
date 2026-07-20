# InstructPix2Pix Pipeline-Stage Attribution

## Final attribution

The observed edit changes are primarily produced by optimized latent delta-z. VAE reconstruction creates a smaller baseline shift. Learned geometry on the original is usually minor; geometry after reconstruction is seed-sensitive and can be amplified by the editor.

## Corrected primary cases

- **Image 1 - add a blue medical mask**: The clean baseline is already an imperfect mask edit. Delta-z changes texture and face coverage, while learned TPS adds little beyond the delta-z replay.
- **Image 2 - add a black jacket**: The clearest transition: latent delta-z produces a seed-sensitive hood-like collapse after 20-30 iterations. Geometry on the original is negligible.
- **Image 4 - add a red scarf**: The requested scarf remains visible. Delta-z mainly changes texture and color; corrected polar geometry modestly changes accessory placement.

## Controls

- Optimization depths: 1, 5, 10, 20, 30 iterations.
- Diffusion seeds: 1234, 24001, 34007 for two cases.
- Loss controls: default, identity term disabled, and larger identity plus preservation weights.
- Neutral geometry and operation-order controls are included.

## Excluded superseded runs

- `pilot3_image1_blue_mask_bspline_10`: Excluded: clean edit used a 256x256 source while ablation stages used a 512x512 normalized source.
- `image1_blue_mask_tps_20_corrected`: Excluded: superseded by corrected_v2 after the same clean-baseline resolution mismatch was fixed.
- `image4_red_scarf_polar_25`: Excluded: zero-state polar warp was not an identity transform; replaced by the corrected polar run.

## Files

- `instruct_pipeline_stage_attribution_report.pdf`
- `valid_primary_attribution_metrics.csv`
- `iteration_sweep.csv`
- `loss_controls.csv`
- `seed_replay_metrics.csv`
