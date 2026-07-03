# SST Benchmark Report

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Shadow resolution: `512`
- Tile size: `128`
- Min leaf size: `1`
- Plane quantization search radius: `0`
- Shadow bias: `0.0015`
- Bake time: `0.005s`
- Readback time: `0.002s`

## Variants

| Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dual_visible | 7.23x | 4.427 | 5.57x | 5.740 | 23504 | 10760 | 0.0118% | 2.3163% | 0.0000% | 2.3163% | 0.0000% | True | True |

## Error Distribution

| Resolution | Variant | Abs err p95 | Abs err p99 | Abs err p99.9 | <=0.5 bias | <=1 bias | <=2 bias | Leak > 1 bias | Shadow > 1 bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | 0.0971% | 0.1499% | 0.1499% | 93.3758% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |

## Dual Layer Utilization

| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |

## Recommendations

| Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Vis mismatch | False lit | False shadow | Leak > 1 bias | Shadow > 1 bias | <=1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_false_shadow_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 5.1992% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 5.1992% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| max_compression_mean_error_le_0_1_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 5.1992% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| max_compression_false_lit_le_1_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 5.1992% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |

## Pareto Front

| Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| dual_visible | 128 | 1 | none | no | 0.001500 | 0.00075 | 0 | 6.68x | 0.0080% | 0.0000% | 2.4752% | 0.0000% | 2.4752% | 0.0000% |
| dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0194% | 0.0000% | 5.1992% | 0.0000% | 5.1992% | 0.0000% |
| dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.00075 | 0 | 8.50x | 0.0180% | 0.0000% | 6.1652% | 0.0000% | 6.1652% | 0.0000% |

## Pareto Visibility Probes

| Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| dual_visible | 128 | 1 | none | no | 0 | 2.4752% | 0.0000% | 2.5265% | 7.3742% | 0.0000% | 0.0000% | 2.4752% |
| dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.1992% | 2.8713% | 6.4827% | 11.4429% | 0.0000% | 0.0000% | 5.1992% |
| dual_relaxed_visible | 128 | 1 | none | no | 0 | 6.1652% | 3.9757% | 7.7187% | 12.9665% | 0.0000% | 0.0000% | 6.1652% |
