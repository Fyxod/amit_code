# Geometry-family attribution summary

This is a separate extension of the verified stage-attribution work. The existing PDF report and its source artifacts were not changed.

## Reading the comparisons

- `output_ssim_vs_clean` compares a stage edit with the clean edit; lower values mean a larger edited-output change, but do not isolate which stage caused it.
- `learned_geometry_vs_neutral_original` isolates learned geometry from neutral grid resampling on the original input.
- `learned_geometry_vs_vae_reconstruction` isolates learned geometry after VAE reconstruction.
- `geometry_increment_after_delta` directly compares the delta-z-only edit with the combined delta-z-plus-geometry edit.
- Input SSIM must be read together with edit SSIM; a large edit change at a visibly damaged input is not subtle geometry.

## Completed cases

- `image2_black_jacket`: 18 runs; prompt `Add a black jacket`
- `image4_red_scarf`: 18 runs; prompt `Add a red scarf`

## Geometry-only ranking on original inputs

### image2_black_jacket

| Geometry | Input SSIM | Edit SSIM vs clean | ArcFace clean vs stage edit |
|---|---:|---:|---:|
| mobius | 0.6965 | 0.6421 | 0.7165 |
| diffgeom | 0.8067 | 0.7490 | 0.8814 |
| homography | 0.8851 | 0.8323 | 0.9380 |
| delaunay | 0.9756 | 0.9100 | 0.9632 |
| bezier | 0.9937 | 0.9515 | 0.9906 |
| bspline | 0.9931 | 0.9555 | 0.9917 |
| laplacian | 0.9937 | 0.9569 | 0.9914 |
| rolling | 0.9921 | 0.9649 | 0.9927 |
| geodesic | 0.9990 | 0.9877 | 0.9937 |
| fft | 0.9994 | 0.9925 | 0.9938 |

### image4_red_scarf

| Geometry | Input SSIM | Edit SSIM vs clean | ArcFace clean vs stage edit |
|---|---:|---:|---:|
| diffgeom | 0.8081 | 0.7589 | 0.6738 |
| mobius | 0.8152 | 0.7812 | 0.9169 |
| homography | 0.8843 | 0.8450 | 0.9343 |
| delaunay | 0.9838 | 0.9440 | 0.9748 |
| rolling | 0.9874 | 0.9594 | 0.9906 |
| bezier | 0.9922 | 0.9636 | 0.9742 |
| bspline | 0.9935 | 0.9664 | 0.9858 |
| laplacian | 0.9935 | 0.9667 | 0.9840 |
| geodesic | 0.9991 | 0.9877 | 0.9958 |
| fft | 0.9992 | 0.9905 | 0.9968 |

## Seed-stability checks

| Case | Run | Direct comparison | Mean edit SSIM | Min | Max | Range |
|---|---|---|---:|---:|---:|---:|
| image2_black_jacket | combined_all | geometry_increment_after_delta | 0.7193 | 0.6620 | 0.8157 | 0.1537 |
| image2_black_jacket | combined_all | learned_geometry_vs_neutral_original | 0.7416 | 0.6506 | 0.7909 | 0.1403 |
| image2_black_jacket | combined_all | learned_geometry_vs_vae_reconstruction | 0.7832 | 0.7684 | 0.8013 | 0.0328 |
| image2_black_jacket | combined_bspline | geometry_increment_after_delta | 0.9061 | 0.7230 | 0.9978 | 0.2748 |
| image2_black_jacket | combined_bspline | learned_geometry_vs_neutral_original | 0.9817 | 0.9775 | 0.9843 | 0.0068 |
| image2_black_jacket | combined_bspline | learned_geometry_vs_vae_reconstruction | 0.9828 | 0.9532 | 0.9977 | 0.0444 |
| image2_black_jacket | combined_bspline_bezier_laplacian | geometry_increment_after_delta | 0.8839 | 0.6591 | 0.9965 | 0.3374 |
| image2_black_jacket | combined_bspline_bezier_laplacian | learned_geometry_vs_neutral_original | 0.9786 | 0.9742 | 0.9814 | 0.0072 |
| image2_black_jacket | combined_bspline_bezier_laplacian | learned_geometry_vs_vae_reconstruction | 0.9757 | 0.9345 | 0.9964 | 0.0618 |
| image2_black_jacket | combined_bspline_tps | geometry_increment_after_delta | 0.8838 | 0.6653 | 0.9939 | 0.3286 |
| image2_black_jacket | combined_bspline_tps | learned_geometry_vs_neutral_original | 0.9738 | 0.9678 | 0.9776 | 0.0098 |
| image2_black_jacket | combined_bspline_tps | learned_geometry_vs_vae_reconstruction | 0.9734 | 0.9324 | 0.9940 | 0.0616 |
| image2_black_jacket | single_bspline | learned_geometry_vs_neutral_original | 0.9643 | 0.9550 | 0.9694 | 0.0144 |
| image2_black_jacket | single_bspline | learned_geometry_vs_vae_reconstruction | 0.9803 | 0.9657 | 0.9886 | 0.0228 |
| image2_black_jacket | single_diffgeom | learned_geometry_vs_neutral_original | 0.7671 | 0.7490 | 0.7830 | 0.0340 |
| image2_black_jacket | single_diffgeom | learned_geometry_vs_vae_reconstruction | 0.6448 | 0.4276 | 0.7629 | 0.3353 |
| image2_black_jacket | single_rolling | learned_geometry_vs_neutral_original | 0.9649 | 0.9623 | 0.9666 | 0.0043 |
| image2_black_jacket | single_rolling | learned_geometry_vs_vae_reconstruction | 0.8377 | 0.5503 | 0.9836 | 0.4332 |
| image4_red_scarf | combined_all | geometry_increment_after_delta | 0.8916 | 0.8815 | 0.8994 | 0.0179 |
| image4_red_scarf | combined_all | learned_geometry_vs_neutral_original | 0.8686 | 0.8661 | 0.8706 | 0.0045 |
| image4_red_scarf | combined_all | learned_geometry_vs_vae_reconstruction | 0.8995 | 0.8940 | 0.9036 | 0.0096 |
| image4_red_scarf | combined_bspline | geometry_increment_after_delta | 0.9974 | 0.9973 | 0.9974 | 0.0001 |
| image4_red_scarf | combined_bspline | learned_geometry_vs_neutral_original | 0.9816 | 0.9812 | 0.9821 | 0.0009 |
| image4_red_scarf | combined_bspline | learned_geometry_vs_vae_reconstruction | 0.9970 | 0.9970 | 0.9971 | 0.0001 |
| image4_red_scarf | combined_bspline_tps | geometry_increment_after_delta | 0.9940 | 0.9938 | 0.9941 | 0.0003 |
| image4_red_scarf | combined_bspline_tps | learned_geometry_vs_neutral_original | 0.9750 | 0.9743 | 0.9756 | 0.0013 |
| image4_red_scarf | combined_bspline_tps | learned_geometry_vs_vae_reconstruction | 0.9934 | 0.9932 | 0.9938 | 0.0006 |
| image4_red_scarf | single_bspline | learned_geometry_vs_neutral_original | 0.9657 | 0.9655 | 0.9661 | 0.0006 |
| image4_red_scarf | single_bspline | learned_geometry_vs_vae_reconstruction | 0.9890 | 0.9886 | 0.9892 | 0.0007 |
| image4_red_scarf | single_diffgeom | learned_geometry_vs_neutral_original | 0.7572 | 0.7544 | 0.7595 | 0.0050 |
| image4_red_scarf | single_diffgeom | learned_geometry_vs_vae_reconstruction | 0.9005 | 0.8983 | 0.9017 | 0.0034 |
| image4_red_scarf | single_rolling | learned_geometry_vs_neutral_original | 0.9577 | 0.9573 | 0.9580 | 0.0007 |
| image4_red_scarf | single_rolling | learned_geometry_vs_vae_reconstruction | 0.9887 | 0.9877 | 0.9895 | 0.0018 |

## Evidence-backed interpretation

- B-spline geometry by itself remains close to the clean edit on the tested original input. The previously observed B-spline-associated failure is therefore not attributable to B-spline alone.
- Large failures after VAE reconstruction are seed-sensitive in the replayed differential-surface and rolling cases: seed 24001 changes sharply, while seeds 1234 and 34007 remain close to normal edits.
- Multi-geometry combinations can redirect or cancel a delta-z-induced failure. This is an interaction effect; it is not evidence that the same geometry produces a stable failure without delta-z.
- Aggressive families such as Mobius, differential-surface deformation, or the all-components combination must be interpreted with their input SSIM because their stronger edit changes can coincide with visible input deformation.

The CSV files beside this document retain all normalized rows and exact image paths for audit.
