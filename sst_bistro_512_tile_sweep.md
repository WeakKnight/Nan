# SST Benchmark Report

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Shadow resolution: `512`
- Tile size: `128`
- Min leaf size: `1`
- Plane quantization search radius: `0`
- Shadow bias: `0.0015`
- Bake time: `0.003s`
- Readback time: `0.002s`

## Variants

| Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dual_visible | 7.23x | 4.427 | 5.57x | 5.740 | 23504 | 10760 | 0.0118% | 2.3163% | 0.0000% | 2.3163% | 0.0000% | True | True |

## Recommendations

| Constraint | Variant | Tile | Min leaf | Plane err | Slack | Q radius | Packed | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_compression_no_false_lit_no_gt_bias_error | dual_visible | 256 | 1 | 0.001000 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| max_compression_vis_mismatch_le_2_5_percent | dual_visible | 256 | 1 | 0.001000 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| max_compression_false_lit_le_1_percent | dual_visible | 256 | 1 | 0.001000 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% |

## Pareto Front

| Variant | Tile | Min leaf | Plane err | Slack | Q radius | Packed | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 256 | 1 | 0.001000 | 0.0015 | 0 | 7.23x | 0.0118% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
