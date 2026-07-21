# Geometry depth and strength controls

These controls extend the 20-iteration family sweep without changing its artifacts.

## image2_black_jacket

| Control | Mode | Stage | Input SSIM | Edit SSIM vs clean | ArcFace clean vs stage edit |
|---|---|---|---:|---:|---:|
| all_iter50_scale05 | combined | learned_combined | 0.7899 | 0.4494 | 0.3611 |
| all_iter50_scale05 | combined | learned_delta_only | 0.7771 | 0.4013 | 0.1705 |
| all_iter50_scale05 | combined | learned_geometry_on_original | 0.9811 | 0.9284 | 0.9803 |
| all_iter50_scale05 | combined | learned_geometry_on_reconstruction | 0.8397 | 0.9372 | 0.9862 |
| all_iter50_scale05 | geometry_only_original | learned_geometry_on_original | 0.7900 | 0.7196 | 0.6986 |
| all_iter50_scale05 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.7340 | 0.5577 | 0.5414 |
| bspline_iter100_scale1 | combined | learned_combined | 0.7873 | 0.4089 | 0.2993 |
| bspline_iter100_scale1 | combined | learned_delta_only | 0.7858 | 0.4020 | 0.2749 |
| bspline_iter100_scale1 | combined | learned_geometry_on_original | 0.9979 | 0.9823 | 0.9930 |
| bspline_iter100_scale1 | combined | learned_geometry_on_reconstruction | 0.8251 | 0.9053 | 0.8857 |
| bspline_iter100_scale1 | geometry_only_original | learned_geometry_on_original | 0.9529 | 0.8931 | 0.9285 |
| bspline_iter100_scale1 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8330 | 0.7155 | 0.7210 |
| bspline_iter100_scale2 | combined | learned_combined | 0.7868 | 0.4065 | 0.3392 |
| bspline_iter100_scale2 | combined | learned_delta_only | 0.7846 | 0.4007 | 0.3296 |
| bspline_iter100_scale2 | combined | learned_geometry_on_original | 0.9978 | 0.9827 | 0.9963 |
| bspline_iter100_scale2 | combined | learned_geometry_on_reconstruction | 0.8259 | 0.9392 | 0.9840 |
| bspline_iter100_scale2 | geometry_only_original | learned_geometry_on_original | 0.9289 | 0.8806 | 0.9114 |
| bspline_iter100_scale2 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8298 | 0.4299 | 0.3208 |
| diffgeom_iter50_scale05 | geometry_only_original | learned_geometry_on_original | 0.8271 | 0.7739 | 0.8316 |
| diffgeom_iter50_scale05 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.7792 | 0.5801 | 0.5820 |
| rolling_iter100_scale1 | geometry_only_original | learned_geometry_on_original | 0.9903 | 0.9589 | 0.9893 |
| rolling_iter100_scale1 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8192 | 0.4258 | 0.3747 |

## image4_red_scarf

| Control | Mode | Stage | Input SSIM | Edit SSIM vs clean | ArcFace clean vs stage edit |
|---|---|---|---:|---:|---:|
| all_iter50_scale05 | combined | learned_combined | 0.8203 | 0.8090 | -0.3060 |
| all_iter50_scale05 | combined | learned_delta_only | 0.8122 | 0.8083 | -0.2825 |
| all_iter50_scale05 | combined | learned_geometry_on_original | 0.9831 | 0.9389 | 0.9796 |
| all_iter50_scale05 | combined | learned_geometry_on_reconstruction | 0.8526 | 0.9375 | 0.9740 |
| all_iter50_scale05 | geometry_only_original | learned_geometry_on_original | 0.7947 | 0.7222 | 0.4022 |
| all_iter50_scale05 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8248 | 0.8437 | 0.8503 |
| bspline_iter100_scale1 | combined | learned_combined | 0.8221 | 0.8019 | -0.3594 |
| bspline_iter100_scale1 | combined | learned_delta_only | 0.8210 | 0.7945 | -0.3712 |
| bspline_iter100_scale1 | combined | learned_geometry_on_original | 0.9977 | 0.9823 | 0.9941 |
| bspline_iter100_scale1 | combined | learned_geometry_on_reconstruction | 0.8417 | 0.9495 | 0.9844 |
| bspline_iter100_scale1 | geometry_only_original | learned_geometry_on_original | 0.9422 | 0.9050 | 0.9397 |
| bspline_iter100_scale1 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8470 | 0.9372 | 0.9728 |
| bspline_iter100_scale2 | combined | learned_combined | 0.8208 | 0.7928 | -0.4040 |
| bspline_iter100_scale2 | combined | learned_delta_only | 0.8193 | 0.7915 | -0.3998 |
| bspline_iter100_scale2 | combined | learned_geometry_on_original | 0.9975 | 0.9833 | 0.9927 |
| bspline_iter100_scale2 | combined | learned_geometry_on_reconstruction | 0.8422 | 0.9494 | 0.9857 |
| bspline_iter100_scale2 | geometry_only_original | learned_geometry_on_original | 0.9296 | 0.8957 | 0.9259 |
| bspline_iter100_scale2 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8432 | 0.9267 | 0.9614 |
| diffgeom_iter50_scale05 | geometry_only_original | learned_geometry_on_original | 0.8473 | 0.8001 | 0.6359 |
| diffgeom_iter50_scale05 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8195 | 0.8845 | 0.8983 |
| rolling_iter100_scale1 | geometry_only_original | learned_geometry_on_original | 0.9388 | 0.8964 | 0.9143 |
| rolling_iter100_scale1 | geometry_only_reconstruction | learned_geometry_on_reconstruction | 0.8357 | 0.9367 | 0.9793 |

## Direct stage increments

| Case | Control | Comparison | Edit SSIM | ArcFace edit similarity |
|---|---|---|---:|---:|
| image2_black_jacket | all_iter50_scale05 | delta_increment_after_reconstruction | 0.4292 | 0.1695 |
| image2_black_jacket | all_iter50_scale05 | geometry_on_original_effect | 0.9284 | 0.9803 |
| image2_black_jacket | all_iter50_scale05 | geometry_on_reconstruction_effect | 0.9169 | 0.8464 |
| image2_black_jacket | all_iter50_scale05 | geometry_increment_after_delta | 0.8641 | 0.6497 |
| image2_black_jacket | all_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | all_iter50_scale05 | geometry_on_original_effect | 0.7196 | 0.6986 |
| image2_black_jacket | all_iter50_scale05 | geometry_on_reconstruction_effect | 0.6063 | 0.6521 |
| image2_black_jacket | all_iter50_scale05 | geometry_increment_after_delta | 0.6063 | 0.6521 |
| image2_black_jacket | all_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | all_iter50_scale05 | geometry_on_original_effect | 0.5589 | 0.5519 |
| image2_black_jacket | all_iter50_scale05 | geometry_on_reconstruction_effect | 0.5444 | 0.3593 |
| image2_black_jacket | all_iter50_scale05 | geometry_increment_after_delta | 0.5444 | 0.3593 |
| image2_black_jacket | bspline_iter100_scale1 | delta_increment_after_reconstruction | 0.4306 | 0.2803 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_on_original_effect | 0.9823 | 0.9930 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_on_reconstruction_effect | 0.9829 | 0.9182 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_increment_after_delta | 0.9912 | 0.9918 |
| image2_black_jacket | bspline_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_on_original_effect | 0.8931 | 0.9285 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_on_reconstruction_effect | 0.8910 | 0.8090 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_increment_after_delta | 0.8910 | 0.8090 |
| image2_black_jacket | bspline_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_on_original_effect | 0.9329 | 0.9687 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_on_reconstruction_effect | 0.7722 | 0.7360 |
| image2_black_jacket | bspline_iter100_scale1 | geometry_increment_after_delta | 0.7722 | 0.7360 |
| image2_black_jacket | bspline_iter100_scale2 | delta_increment_after_reconstruction | 0.4288 | 0.2941 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_on_original_effect | 0.9827 | 0.9963 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_on_reconstruction_effect | 0.9491 | 0.8603 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_increment_after_delta | 0.9902 | 0.9815 |
| image2_black_jacket | bspline_iter100_scale2 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_on_original_effect | 0.8806 | 0.9114 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_on_reconstruction_effect | 0.8790 | 0.7685 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_increment_after_delta | 0.8790 | 0.7685 |
| image2_black_jacket | bspline_iter100_scale2 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_on_original_effect | 0.9227 | 0.9610 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_on_reconstruction_effect | 0.4665 | 0.3086 |
| image2_black_jacket | bspline_iter100_scale2 | geometry_increment_after_delta | 0.4665 | 0.3086 |
| image2_black_jacket | diffgeom_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | diffgeom_iter50_scale05 | geometry_on_original_effect | 0.7739 | 0.8316 |
| image2_black_jacket | diffgeom_iter50_scale05 | geometry_on_reconstruction_effect | 0.7732 | 0.8250 |
| image2_black_jacket | diffgeom_iter50_scale05 | geometry_increment_after_delta | 0.7732 | 0.8250 |
| image2_black_jacket | diffgeom_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | diffgeom_iter50_scale05 | geometry_on_original_effect | 0.8167 | 0.8234 |
| image2_black_jacket | diffgeom_iter50_scale05 | geometry_on_reconstruction_effect | 0.6125 | 0.6023 |
| image2_black_jacket | diffgeom_iter50_scale05 | geometry_increment_after_delta | 0.6125 | 0.6023 |
| image2_black_jacket | rolling_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | rolling_iter100_scale1 | geometry_on_original_effect | 0.9589 | 0.9893 |
| image2_black_jacket | rolling_iter100_scale1 | geometry_on_reconstruction_effect | 0.9729 | 0.9530 |
| image2_black_jacket | rolling_iter100_scale1 | geometry_increment_after_delta | 0.9729 | 0.9530 |
| image2_black_jacket | rolling_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image2_black_jacket | rolling_iter100_scale1 | geometry_on_original_effect | 0.9199 | 0.9701 |
| image2_black_jacket | rolling_iter100_scale1 | geometry_on_reconstruction_effect | 0.4641 | 0.3500 |
| image2_black_jacket | rolling_iter100_scale1 | geometry_increment_after_delta | 0.4641 | 0.3500 |
| image4_red_scarf | all_iter50_scale05 | delta_increment_after_reconstruction | 0.8133 | -0.2366 |
| image4_red_scarf | all_iter50_scale05 | geometry_on_original_effect | 0.9389 | 0.9796 |
| image4_red_scarf | all_iter50_scale05 | geometry_on_reconstruction_effect | 0.9750 | 0.9898 |
| image4_red_scarf | all_iter50_scale05 | geometry_increment_after_delta | 0.9795 | 0.9918 |
| image4_red_scarf | all_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | all_iter50_scale05 | geometry_on_original_effect | 0.7222 | 0.4022 |
| image4_red_scarf | all_iter50_scale05 | geometry_on_reconstruction_effect | 0.7341 | 0.3708 |
| image4_red_scarf | all_iter50_scale05 | geometry_increment_after_delta | 0.7341 | 0.3708 |
| image4_red_scarf | all_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | all_iter50_scale05 | geometry_on_original_effect | 0.8212 | 0.8639 |
| image4_red_scarf | all_iter50_scale05 | geometry_on_reconstruction_effect | 0.8572 | 0.8667 |
| image4_red_scarf | all_iter50_scale05 | geometry_increment_after_delta | 0.8572 | 0.8667 |
| image4_red_scarf | bspline_iter100_scale1 | delta_increment_after_reconstruction | 0.7911 | -0.3610 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_on_original_effect | 0.9823 | 0.9941 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_on_reconstruction_effect | 0.9978 | 0.9993 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_increment_after_delta | 0.9907 | 0.9817 |
| image4_red_scarf | bspline_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_on_original_effect | 0.9050 | 0.9397 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_on_reconstruction_effect | 0.9473 | 0.9437 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_increment_after_delta | 0.9473 | 0.9437 |
| image4_red_scarf | bspline_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_on_original_effect | 0.9416 | 0.9754 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_on_reconstruction_effect | 0.9732 | 0.9846 |
| image4_red_scarf | bspline_iter100_scale1 | geometry_increment_after_delta | 0.9732 | 0.9846 |
| image4_red_scarf | bspline_iter100_scale2 | delta_increment_after_reconstruction | 0.7886 | -0.3820 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_on_original_effect | 0.9833 | 0.9927 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_on_reconstruction_effect | 0.9971 | 0.9988 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_increment_after_delta | 0.9977 | 0.9985 |
| image4_red_scarf | bspline_iter100_scale2 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_on_original_effect | 0.8957 | 0.9259 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_on_reconstruction_effect | 0.9351 | 0.9356 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_increment_after_delta | 0.9351 | 0.9356 |
| image4_red_scarf | bspline_iter100_scale2 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_on_original_effect | 0.9197 | 0.9777 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_on_reconstruction_effect | 0.9586 | 0.9759 |
| image4_red_scarf | bspline_iter100_scale2 | geometry_increment_after_delta | 0.9586 | 0.9759 |
| image4_red_scarf | diffgeom_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | diffgeom_iter50_scale05 | geometry_on_original_effect | 0.8001 | 0.6359 |
| image4_red_scarf | diffgeom_iter50_scale05 | geometry_on_reconstruction_effect | 0.8261 | 0.6111 |
| image4_red_scarf | diffgeom_iter50_scale05 | geometry_increment_after_delta | 0.8261 | 0.6111 |
| image4_red_scarf | diffgeom_iter50_scale05 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | diffgeom_iter50_scale05 | geometry_on_original_effect | 0.8777 | 0.8980 |
| image4_red_scarf | diffgeom_iter50_scale05 | geometry_on_reconstruction_effect | 0.9098 | 0.9009 |
| image4_red_scarf | diffgeom_iter50_scale05 | geometry_increment_after_delta | 0.9098 | 0.9009 |
| image4_red_scarf | rolling_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | rolling_iter100_scale1 | geometry_on_original_effect | 0.8964 | 0.9143 |
| image4_red_scarf | rolling_iter100_scale1 | geometry_on_reconstruction_effect | 0.9291 | 0.9282 |
| image4_red_scarf | rolling_iter100_scale1 | geometry_increment_after_delta | 0.9291 | 0.9282 |
| image4_red_scarf | rolling_iter100_scale1 | delta_increment_after_reconstruction | 1.0000 | 1.0000 |
| image4_red_scarf | rolling_iter100_scale1 | geometry_on_original_effect | 0.9591 | 0.9919 |
| image4_red_scarf | rolling_iter100_scale1 | geometry_on_reconstruction_effect | 0.9793 | 0.9926 |
| image4_red_scarf | rolling_iter100_scale1 | geometry_increment_after_delta | 0.9793 | 0.9926 |

Interpret edit changes together with input SSIM. Controls with severe input damage are not evidence of an imperceptible geometric effect.
