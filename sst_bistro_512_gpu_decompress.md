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

## GPU Decompression

This validates the paper-style Compact SST stream by decompressing it on the GPU back to an `r32_float` depth texture.

| Resolution | Variant | GPU == CPU | GPU vs CPU mean | GPU vs CPU max | GPU vs source mean | GPU vs source RMSE | GPU vs source max | Dispatch | Readback |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | True | 0.0000% | 0.0000% | 0.0118% | 0.0336% | 0.1499% | 0.003s | 0.001s |

## Dual Layer Utilization

| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |
