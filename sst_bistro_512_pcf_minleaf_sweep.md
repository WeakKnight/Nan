# SST Benchmark Report

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Shadow resolution: `512`
- Tile size: `128`
- Min leaf size: `1`
- Plane quantization search radius: `0`
- Shadow bias: `0.0015`
- Bake time: `0.004s`
- Readback time: `0.002s`

## Variants

| Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dual_visible | 7.23x | 4.427 | 5.57x | 5.740 | 23504 | 10760 | 0.0118% | 2.3163% | 0.0000% | 2.3163% | 0.0000% | True | True |

## Error Distribution

| Resolution | Variant | Abs err p95 | Abs err p99 | Abs err p99.9 | <=0.5 bias | <=1 bias | <=2 bias | Leak > 1 bias | Shadow > 1 bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | 0.0971% | 0.1499% | 0.1499% | 93.3758% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |

## Depth Texture PCF3 Delta

This estimates the shader-side `Compact SST PCF3` mode against the existing `Depth Texture` PCF3 path.

| Resolution | Variant | SST PCF3 vs Depth PCF3 MAE | SST PCF3 max | SST hard vs Depth PCF3 MAE | SST hard max | @0B hard MAE | @1B hard MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | 1.7765% | 77.7778% | 13.6111% | 100.0000% | 10.1309% | 18.8372% |

## Dual Layer Utilization

| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |

## Recommendations

| Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Vis mismatch | PCF3 MAE | False lit | False shadow | Leak > 1 bias | Shadow > 1 bias | <=1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 8.68x | 3.1948% | 2.1255% | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 8.68x | 3.1948% | 2.1255% | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 5.9100% | 3.2294% | 0.0000% | 5.9100% | 0.0000% | 3.8025% | 96.1975% |
| max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 5.9100% | 3.2294% | 0.0000% | 5.9100% | 0.0000% | 3.8025% | 96.1975% |
| max_compression_mean_error_le_0_1_percent | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 10.0773% | 5.8398% | 0.0000% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| max_compression_false_lit_le_1_percent | dual_loose_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 56.64x | 20.6970% | 14.2423% | 0.0025% | 20.6945% | 0.0099% | 25.2102% | 74.7799% |
| max_compression_pcf3_mae_le_2_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_pcf3_mae_le_3_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 5.1992% | 2.9317% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| max_compression_pcf3_mae_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 5.9100% | 3.2294% | 0.0000% | 5.9100% | 0.0000% | 3.8025% | 96.1975% |

## Pareto Front

| Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 8.68x | 0.0907% | 2.3087% | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0194% | 0.0000% | 5.1992% | 0.0000% | 5.1992% | 0.0000% |
| dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 0.0976% | 1.8616% | 5.9100% | 0.0000% | 5.9100% | 0.0000% |
| dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 0.0409% | 0.0000% | 10.0773% | 0.0000% | 10.0773% | 0.0000% |
| dual_loose_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 13.16x | 0.1184% | 1.5427% | 10.6684% | 0.0000% | 10.6684% | 0.0000% |
| dual_visible | 128 | 4 | none | no | 0.001500 | 0.0015 | 0 | 18.52x | 0.8184% | 15.3687% | 9.8669% | 0.0006% | 9.8663% | 0.0023% |
| dual_relaxed_visible | 128 | 4 | none | no | 0.001500 | 0.0015 | 0 | 19.97x | 0.8219% | 12.1887% | 11.2012% | 0.0006% | 11.2006% | 0.0023% |
| dual_loose_visible | 128 | 4 | none | no | 0.001500 | 0.0015 | 0 | 24.22x | 0.8383% | 9.7717% | 14.9347% | 0.0006% | 14.9342% | 0.0023% |
| dual_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 52.90x | 2.1083% | 29.1992% | 19.5634% | 0.0025% | 19.5609% | 0.0099% |
| dual_relaxed_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 54.23x | 2.1095% | 26.8555% | 19.9996% | 0.0025% | 19.9971% | 0.0099% |
| dual_loose_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 56.64x | 2.1127% | 22.2412% | 20.6970% | 0.0025% | 20.6945% | 0.0099% |

## Pareto Visibility Probes

| Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| dual_visible | 128 | 2 | none | no | 0 | 3.1948% | 1.1528% | 3.4721% | 8.1543% | 0.0000% | 0.0000% | 3.1948% |
| dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.1992% | 2.8713% | 6.4827% | 11.4429% | 0.0000% | 0.0000% | 5.1992% |
| dual_relaxed_visible | 128 | 2 | none | no | 0 | 5.9100% | 3.8021% | 7.4135% | 12.4245% | 0.0000% | 0.0000% | 5.9100% |
| dual_loose_visible | 128 | 1 | none | no | 0 | 10.0773% | 9.1370% | 13.3266% | 17.8455% | 0.0000% | 0.0000% | 10.0773% |
| dual_loose_visible | 128 | 2 | none | no | 0 | 10.6684% | 9.9083% | 14.0980% | 18.6672% | 0.0000% | 0.0000% | 10.6684% |
| dual_visible | 128 | 4 | none | no | 0 | 9.8669% | 10.3977% | 12.5111% | 16.5565% | 0.0023% | 0.0006% | 9.8663% |
| dual_relaxed_visible | 128 | 4 | none | no | 0 | 11.2012% | 11.7046% | 14.4775% | 18.6203% | 0.0023% | 0.0006% | 11.2006% |
| dual_loose_visible | 128 | 4 | none | no | 0 | 14.9347% | 16.3986% | 19.7819% | 23.5561% | 0.0023% | 0.0006% | 14.9342% |
| dual_visible | 128 | 8 | none | no | 0 | 19.5634% | 23.7091% | 25.8389% | 28.6957% | 0.0099% | 0.0025% | 19.5609% |
| dual_relaxed_visible | 128 | 8 | none | no | 0 | 19.9996% | 24.2519% | 26.4523% | 29.2843% | 0.0099% | 0.0025% | 19.9971% |
| dual_loose_visible | 128 | 8 | none | no | 0 | 20.6970% | 25.2090% | 27.4078% | 30.1613% | 0.0099% | 0.0025% | 20.6945% |
