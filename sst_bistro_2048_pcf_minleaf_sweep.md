# SST Benchmark Report

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Shadow resolution: `2048`
- Tile size: `128`
- Min leaf size: `1`
- Plane quantization search radius: `0`
- Shadow bias: `0.0015`
- Bake time: `0.004s`
- Readback time: `0.019s`

## Variants

| Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dual_visible | 22.67x | 1.412 | 16.37x | 1.955 | 128004 | 71251 | 0.0112% | 2.3436% | 0.0000% | 2.3436% | 0.0000% | True | True |

## Error Distribution

| Resolution | Variant | Abs err p95 | Abs err p99 | Abs err p99.9 | <=0.5 bias | <=1 bias | <=2 bias | Leak > 1 bias | Shadow > 1 bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | dual_visible | 0.0818% | 0.1361% | 0.1499% | 94.4060% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |

## Depth Texture PCF3 Delta

This estimates the shader-side `Compact SST PCF3` mode against the existing `Depth Texture` PCF3 path.

| Resolution | Variant | SST PCF3 vs Depth PCF3 MAE | SST PCF3 max | SST hard vs Depth PCF3 MAE | SST hard max | @0B hard MAE | @1B hard MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2048 | dual_visible | 2.3787% | 100.0000% | 6.2393% | 100.0000% | 0.9921% | 18.7816% |

## Dual Layer Utilization

| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | dual_visible | 38.3317% | 3.5489% | 14.1332% | 89.0627% | 0.0579% | 0.1500% | 0.1500% | 36.1833% |

## Memory Breakdown

Decomp working set counts the persistent Compact SST stream plus one full-resolution `r32_float` decompressed depth texture.

| Resolution | Variant | Packed bytes | Packed ratio | Packed stream bpt | Tile roots bpt | Packed total bpt | Decomp texture bytes | Packed+decomp bytes | Packed+decomp bpt | Packed+decomp ratio | Fixed64 bytes | Fixed64 ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | dual_visible | 740,052 | 22.67x | 1.410 | 0.002 | 1.412 | 16,777,216 | 17,517,268 | 33.412 | 0.96x | 1,025,056 | 16.37x |

## Recommendations

| Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Packed bpt | Packed+decomp ratio | Vis mismatch | PCF3 MAE | False lit | False shadow | Leak > 1 bias | Shadow > 1 bias | <=1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 22.67x | 1.412 | 0.96x | 2.3436% | 2.3787% | 0.0000% | 2.3436% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 22.67x | 1.412 | 0.96x | 2.3436% | 2.3787% | 0.0000% | 2.3436% | 0.0000% | 0.0000% | 100.0000% |
| max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |
| max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |
| max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| max_compression_mean_error_le_0_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| max_compression_false_lit_le_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| max_compression_pcf3_mae_le_3_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |
| max_compression_pcf3_mae_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |

## Pareto Front

| Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 22.67x | 0.0112% | 0.0000% | 2.3436% | 0.0000% | 2.3436% | 0.0000% |
| dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 0.0322% | 0.7489% | 2.6270% | 0.0000% | 2.6270% | 0.0000% |
| dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 28.80x | 0.0203% | 0.0000% | 5.8971% | 0.0000% | 5.8971% | 0.0000% |
| dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.0412% | 0.6092% | 6.1282% | 0.0000% | 6.1282% | 0.0000% |

## Pareto Visibility Probes

| Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | none | no | 0 | 2.3436% | 0.0000% | 1.6765% | 7.6977% | 0.0000% | 0.0000% | 2.3436% |
| dual_visible | 128 | 2 | none | no | 0 | 2.6270% | 0.3737% | 2.0510% | 8.0831% | 0.0000% | 0.0000% | 2.6270% |
| dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.8971% | 3.4296% | 7.0563% | 13.1024% | 0.0000% | 0.0000% | 5.8971% |
| dual_relaxed_visible | 128 | 2 | none | no | 0 | 6.1282% | 3.7342% | 7.3609% | 13.4178% | 0.0000% | 0.0000% | 6.1282% |
