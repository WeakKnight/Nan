# SST Benchmark Report

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Shadow resolution: `1024`
- Tile size: `128`
- Min leaf size: `1`
- Plane quantization search radius: `0`
- Shadow bias: `0.0015`
- Bake time: `0.004s`
- Readback time: `0.006s`

## Variants

| Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dual_visible | 12.60x | 2.541 | 9.47x | 3.381 | 55360 | 27535 | 0.0109% | 2.1022% | 0.0000% | 2.1022% | 0.0000% | True | True |

## Error Distribution

| Resolution | Variant | Abs err p95 | Abs err p99 | Abs err p99.9 | <=0.5 bias | <=1 bias | <=2 bias | Leak > 1 bias | Shadow > 1 bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | dual_visible | 0.0819% | 0.1345% | 0.1499% | 94.3095% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |

## Depth Texture PCF3 Delta

This estimates the shader-side `Compact SST PCF3` mode against the existing `Depth Texture` PCF3 path.

| Resolution | Variant | SST PCF3 vs Depth PCF3 MAE | SST PCF3 max | SST hard vs Depth PCF3 MAE | SST hard max | @0B hard MAE | @1B hard MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1024 | dual_visible | 2.1657% | 100.0000% | 8.9291% | 100.0000% | 3.1407% | 18.7047% |

## Dual Layer Utilization

| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | dual_visible | 38.3643% | 3.5414% | 14.1052% | 89.0352% | 0.0578% | 0.1500% | 0.1500% | 36.1257% |

## Recommendations

| Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Vis mismatch | PCF3 MAE | False lit | False shadow | Leak > 1 bias | Shadow > 1 bias | <=1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 12.60x | 2.1022% | 2.1657% | 0.0000% | 2.1022% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 12.60x | 2.1022% | 2.1657% | 0.0000% | 2.1022% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 2.5915% | 2.3723% | 0.0000% | 2.5915% | 0.0000% | 0.6459% | 99.3541% |
| max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 2.5915% | 2.3723% | 0.0000% | 2.5915% | 0.0000% | 0.6459% | 99.3541% |
| max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| max_compression_mean_error_le_0_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| max_compression_false_lit_le_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| max_compression_pcf3_mae_le_3_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 2.5915% | 2.3723% | 0.0000% | 2.5915% | 0.0000% | 0.6459% | 99.3541% |
| max_compression_pcf3_mae_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |

## Pareto Front

| Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 12.60x | 0.0109% | 0.0000% | 2.1022% | 0.0000% | 2.1022% | 0.0000% |
| dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 0.0509% | 1.2932% | 2.5915% | 0.0000% | 2.5915% | 0.0000% |
| dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 15.87x | 0.0200% | 0.0000% | 5.5929% | 0.0000% | 5.5929% | 0.0000% |
| dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 0.0597% | 1.0532% | 5.9924% | 0.0000% | 5.9924% | 0.0000% |

## Pareto Visibility Probes

| Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0 | 2.1022% | 0.0000% | 1.6140% | 6.7947% | 0.0000% | 0.0000% | 2.1022% |
| dual_visible | 128 | 2 | none | no | 0 | 2.5915% | 0.6459% | 2.2606% | 7.4596% | 0.0000% | 0.0000% | 2.5915% |
| dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.5929% | 3.2820% | 6.8816% | 12.2081% | 0.0000% | 0.0000% | 5.5929% |
| dual_relaxed_visible | 128 | 2 | none | no | 0 | 5.9924% | 3.8086% | 7.4082% | 12.7526% | 0.0000% | 0.0000% | 5.9924% |
