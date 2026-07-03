# SST Benchmark Report

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Shadow resolution: `512`
- Tile size: `128`
- Min leaf size: `1`
- Shadow bias: `0.0015`
- Bake time: `0.005s`
- Readback time: `0.002s`

## Variants

| Variant | Packed | Fixed64 | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_visible | 7.23x | 5.57x | 23504 | 10760 | 0.0118% | 2.3163% | 0.0000% | 2.3163% | 0.0000% | True |

## Recommendations

| Constraint | Variant | Plane err | Slack | Packed | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| max_compression_no_false_lit_no_gt_bias_error | dual_visible | 0.001000 | 0.0015 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| max_compression_vis_mismatch_le_1_percent | dual_half_visible | 0.001000 | 0.0015 | 6.11x | 0.8914% | 0.0000% | 0.8914% | 0.0000% |
| max_compression_vis_mismatch_le_2_5_percent | dual_visible | 0.001000 | 0.0015 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| max_compression_false_lit_le_1_percent | dual_capped | 0.001500 | 0.0015 | 8.80x | 3.3482% | 0.9258% | 2.4223% | 3.7033% |

## Pareto Front

| Variant | Plane err | Slack | Packed | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_bias | 0.001000 | 0.0015 | 4.87x | 0.0036% | 0.0036% | 0.0000% | 0.0036% | 0.0000% |
| dual_half_visible | 0.001000 | 0.0015 | 6.11x | 0.0073% | 0.8914% | 0.0000% | 0.8914% | 0.0000% |
| dual_visible | 0.001000 | 0.0015 | 7.23x | 0.0118% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| dual_capped | 0.001000 | 0.0015 | 7.85x | 0.0144% | 2.4583% | 0.7122% | 1.7461% | 2.8488% |
| dual_capped | 0.001500 | 0.0015 | 8.80x | 0.0181% | 3.3482% | 0.9258% | 2.4223% | 3.7033% |
| dual_capped | 0.001000 | 0.003 | 8.23x | 0.0193% | 2.5042% | 1.1323% | 1.3719% | 4.5288% |
| dual_capped | 0.002500 | 0.0015 | 10.33x | 0.0262% | 5.1989% | 1.3771% | 3.8218% | 5.5084% |
| dual_capped | 0.001500 | 0.003 | 9.18x | 0.0241% | 3.3706% | 1.3948% | 1.9757% | 5.5790% |
| dual_capped | 0.002500 | 0.003 | 10.74x | 0.0370% | 5.6034% | 2.0377% | 3.5657% | 8.1509% |
