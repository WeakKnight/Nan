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

## Recommendations

| Constraint | Variant | Tile | Min leaf | Plane err | Slack | Q radius | Packed | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | 0.001000 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | 0.001000 | 0.0015 | 0 | 7.23x | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | 0.001000 | 0.0015 | 0 | 8.68x | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| max_compression_mean_error_le_0_1_percent | dual_visible | 128 | 2 | 0.001000 | 0.0015 | 0 | 8.68x | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| max_compression_false_lit_le_1_percent | dual_visible | 128 | 8 | 0.001000 | 0.0015 | 0 | 52.90x | 19.5634% | 0.0025% | 19.5609% | 0.0099% |

## Pareto Front

| Variant | Tile | Min leaf | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | 0.001000 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| dual_visible | 128 | 2 | 0.001000 | 0.0015 | 0 | 8.68x | 0.0907% | 2.3087% | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| dual_visible | 128 | 4 | 0.001000 | 0.0015 | 0 | 18.52x | 0.8184% | 15.3687% | 9.8669% | 0.0006% | 9.8663% | 0.0023% |
| dual_visible | 128 | 8 | 0.001000 | 0.0015 | 0 | 52.90x | 2.1083% | 29.1992% | 19.5634% | 0.0025% | 19.5609% | 0.0099% |

## Pareto Visibility Probes

| Variant | Tile | Min leaf | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_visible | 128 | 1 | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| dual_visible | 128 | 2 | 0 | 3.1948% | 1.1528% | 3.4721% | 8.1543% | 0.0000% | 0.0000% | 3.1948% |
| dual_visible | 128 | 4 | 0 | 9.8669% | 10.3977% | 12.5111% | 16.5565% | 0.0023% | 0.0006% | 9.8663% |
| dual_visible | 128 | 8 | 0 | 19.5634% | 23.7091% | 25.8389% | 28.6957% | 0.0099% | 0.0025% | 19.5609% |
