# Saved cases

This folder contains visually inspected prompt-sensitivity examples. The
prompt and numerical metadata in each case folder are copied directly from the
corresponding deterministic paired-edit run. Run `materialize_parth_save.py`
to reproduce the curated folders from `selection_manifest.json`, then run
`evaluate_parth_save_arcface.py` on the A6000 to add exact ArcFace metrics.

`image_2_black_jacket_igs2_seed24001/` preserves the selected corrected result with its actual prompt, editor settings, metrics, and five source artifacts.

The experiment prompt was `Add a black jacket`. The output also darkened the hair; that is an observed side effect rather than the recorded prompt.
