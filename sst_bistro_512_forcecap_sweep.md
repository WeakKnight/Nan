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

## Recommendations

| Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| max_compression_mean_error_le_0_1_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| max_compression_false_lit_le_1_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.1948% | 0.0000% | 3.1948% | 0.0000% |

## Pareto Front

| Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 0.0907% | 2.3087% | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| dual_visible | 128 | 2 | 0.010000 | no | 0.001000 | 0.0015 | 0 | 7.77x | 0.5754% | 0.9567% | 2.9521% | 0.2768% | 2.6753% | 1.1070% |
| dual_visible | 128 | 2 | 0.005000 | no | 0.001000 | 0.0015 | 0 | 7.64x | 0.6828% | 0.7431% | 2.9251% | 0.3300% | 2.5951% | 1.3199% |

## Pareto Visibility Probes

| Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 2 | none | no | 0 | 3.1948% | 1.1528% | 3.4721% | 8.1543% | 0.0000% | 0.0000% | 3.1948% |
| dual_visible | 128 | 2 | 0.010000 | no | 0 | 2.9521% | 0.4768% | 2.7962% | 7.4284% | 1.1070% | 0.2768% | 2.6753% |
| dual_visible | 128 | 2 | 0.005000 | no | 0 | 2.9251% | 0.3700% | 2.6894% | 7.3212% | 1.3199% | 0.3300% | 2.5951% |
