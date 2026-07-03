# SST Benchmark Comparison

- Scenes: `E:\GitHub\niagara_bistro\bistro.gltf`
- Sources: `sst_bistro_512.json`, `sst_bistro_512_forcecap_sweep.json`, `sst_bistro_512_gpu_decompress.json`, `sst_bistro_512_memory_metrics.json`, `sst_bistro_512_minleaf_sweep.json`, `sst_bistro_512_pcf_minleaf_sweep.json`, `sst_bistro_512_pcf_profile_sweep.json`, `sst_bistro_512_profile_sweep.json`, `sst_bistro_512_qradius_sweep.json`, `sst_bistro_512_slack_sweep.json`, `sst_bistro_512_sweep.json`, `sst_bistro_512_tile_sweep.json`, `sst_bistro_1024.json`, `sst_bistro_1024_pcf_minleaf_sweep.json`, `sst_bistro_2048.json`, `sst_bistro_2048_pcf_minleaf_sweep.json`
- Tile size: `128`
- Min leaf size: `1`
- Plane quantization search radius: `0`
- Shadow bias: `0.0015`

## Resolution / Variant Matrix

| Resolution | Variant | Packed | Packed bpt | Fixed64 | Fixed64 bpt | Nodes | 30-bit leaves | Mean depth err | Vis mismatch | False lit | False shadow | Leak > 1 bias | Morton parity | Fixed64 parity |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 512 | dual_bias | 4.87x | 6.574 | 3.76x | 8.504 | 34824 | 15808 | 0.0036% | 0.0036% | 0.0000% | 0.0036% | 0.0000% | True | True |
| 512 | dual_half_visible | 6.11x | 5.233 | 4.84x | 6.615 | 27088 | 11320 | 0.0073% | 0.8914% | 0.0000% | 0.8914% | 0.0000% | True | True |
| 512 | dual_visible | 7.23x | 4.427 | 5.57x | 5.740 | 23504 | 10760 | 0.0118% | 2.3163% | 0.0000% | 2.3163% | 0.0000% | True | True |
| 512 | dual_relaxed_visible | 8.80x | 3.638 | 6.62x | 4.837 | 19804 | 9822 | 0.0194% | 5.1992% | 0.0000% | 5.1992% | 0.0000% | True | True |
| 512 | dual_loose_visible | 10.94x | 2.925 | 7.92x | 4.038 | 16532 | 9120 | 0.0409% | 10.0773% | 0.0000% | 10.0773% | 0.0000% | True | True |
| 512 | dual_capped | 8.80x | 3.638 | 6.62x | 4.837 | 19804 | 9822 | 0.0181% | 3.3482% | 0.9258% | 2.4223% | 3.7033% | True | True |
| 512 | single | 5.77x | 5.546 | 4.69x | 6.829 | 27964 | 10512 | 0.0052% | 9.4118% | 0.0000% | 9.4118% | 0.0000% | True | True |
| 512 | dual_safe | 5.77x | 5.546 | 4.69x | 6.829 | 27964 | 10512 | 0.0052% | 9.4118% | 0.0000% | 9.4118% | 0.0000% | True | True |
| 512 | dual_raw | 10.84x | 2.951 | 7.70x | 4.158 | 17024 | 9888 | 0.9946% | 6.3375% | 4.6355% | 1.7020% | 18.5421% | True | True |
| 1024 | dual_bias | 7.79x | 4.109 | 5.81x | 5.506 | 90172 | 45758 | 0.0057% | 0.0065% | 0.0000% | 0.0065% | 0.0000% | True | True |
| 1024 | dual_half_visible | 10.64x | 3.006 | 8.09x | 3.957 | 64804 | 31165 | 0.0082% | 0.9603% | 0.0000% | 0.9603% | 0.0000% | True | True |
| 1024 | dual_visible | 12.60x | 2.541 | 9.47x | 3.381 | 55360 | 27535 | 0.0109% | 2.1022% | 0.0000% | 2.1022% | 0.0000% | True | True |
| 1024 | dual_relaxed_visible | 15.87x | 2.016 | 11.57x | 2.765 | 45276 | 24558 | 0.0200% | 5.5929% | 0.0000% | 5.5929% | 0.0000% | True | True |
| 1024 | dual_loose_visible | 19.77x | 1.618 | 13.99x | 2.288 | 37456 | 21949 | 0.0391% | 10.1970% | 0.0000% | 10.1970% | 0.0000% | True | True |
| 1024 | dual_capped | 15.87x | 2.016 | 11.57x | 2.765 | 45272 | 24555 | 0.0181% | 3.1276% | 0.6797% | 2.4479% | 2.7185% | True | True |
| 1024 | single | 10.43x | 3.067 | 8.09x | 3.958 | 64812 | 29184 | 0.0076% | 10.4506% | 0.0000% | 10.4506% | 0.0000% | True | True |
| 1024 | dual_safe | 10.43x | 3.067 | 8.09x | 3.958 | 64812 | 29184 | 0.0076% | 10.4506% | 0.0000% | 10.4506% | 0.0000% | True | True |
| 1024 | dual_raw | 19.09x | 1.676 | 13.39x | 2.389 | 39116 | 23364 | 0.9683% | 6.2991% | 4.4334% | 1.8657% | 17.7333% | True | True |
| 2048 | dual_bias | 12.49x | 2.562 | 9.16x | 3.495 | 228900 | 122214 | 0.0054% | 0.0122% | 0.0000% | 0.0122% | 0.0000% | True | True |
| 2048 | dual_half_visible | 18.62x | 1.719 | 13.59x | 2.355 | 154228 | 83400 | 0.0085% | 1.1391% | 0.0000% | 1.1391% | 0.0000% | True | True |
| 2048 | dual_visible | 22.67x | 1.412 | 16.37x | 1.955 | 128004 | 71251 | 0.0112% | 2.3436% | 0.0000% | 2.3436% | 0.0000% | True | True |
| 2048 | dual_relaxed_visible | 28.80x | 1.111 | 20.32x | 1.575 | 103068 | 60739 | 0.0203% | 5.8971% | 0.0000% | 5.8971% | 0.0000% | True | True |
| 2048 | dual_loose_visible | 35.89x | 0.892 | 24.77x | 1.292 | 84532 | 52443 | 0.0398% | 10.6017% | 0.0000% | 10.6017% | 0.0000% | True | True |
| 2048 | dual_capped | 28.80x | 1.111 | 20.32x | 1.575 | 103064 | 60738 | 0.0183% | 3.1643% | 0.7286% | 2.4356% | 2.9145% | True | True |
| 2048 | single | 18.57x | 1.723 | 14.01x | 2.284 | 149572 | 73590 | 0.0075% | 10.6878% | 0.0000% | 10.6878% | 0.0000% | True | True |
| 2048 | dual_safe | 18.57x | 1.723 | 14.01x | 2.284 | 149572 | 73590 | 0.0075% | 10.6878% | 0.0000% | 10.6878% | 0.0000% | True | True |
| 2048 | dual_raw | 34.24x | 0.934 | 23.65x | 1.353 | 88544 | 54864 | 0.9750% | 6.3583% | 4.5233% | 1.8350% | 18.0931% | True | True |

## Profile Trend Summary

| Resolution | Variant | Packed | Packed bpt | Mean depth err | Vis mismatch | False lit | Leak > 1 bias |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 4.87x | 6.574 | 0.0036% | 0.0036% | 0.0000% | 0.0000% |
| 512 | dual_half_visible | 6.11x | 5.233 | 0.0073% | 0.8914% | 0.0000% | 0.0000% |
| 512 | dual_visible | 7.23x | 4.427 | 0.0118% | 2.3163% | 0.0000% | 0.0000% |
| 512 | dual_relaxed_visible | 8.80x | 3.638 | 0.0194% | 5.1992% | 0.0000% | 0.0000% |
| 512 | dual_loose_visible | 10.94x | 2.925 | 0.0409% | 10.0773% | 0.0000% | 0.0000% |
| 512 | dual_capped | 8.80x | 3.638 | 0.0181% | 3.3482% | 0.9258% | 3.7033% |
| 1024 | dual_bias | 7.79x | 4.109 | 0.0057% | 0.0065% | 0.0000% | 0.0000% |
| 1024 | dual_half_visible | 10.64x | 3.006 | 0.0082% | 0.9603% | 0.0000% | 0.0000% |
| 1024 | dual_visible | 12.60x | 2.541 | 0.0109% | 2.1022% | 0.0000% | 0.0000% |
| 1024 | dual_relaxed_visible | 15.87x | 2.016 | 0.0200% | 5.5929% | 0.0000% | 0.0000% |
| 1024 | dual_loose_visible | 19.77x | 1.618 | 0.0391% | 10.1970% | 0.0000% | 0.0000% |
| 1024 | dual_capped | 15.87x | 2.016 | 0.0181% | 3.1276% | 0.6797% | 2.7185% |
| 2048 | dual_bias | 12.49x | 2.562 | 0.0054% | 0.0122% | 0.0000% | 0.0000% |
| 2048 | dual_half_visible | 18.62x | 1.719 | 0.0085% | 1.1391% | 0.0000% | 0.0000% |
| 2048 | dual_visible | 22.67x | 1.412 | 0.0112% | 2.3436% | 0.0000% | 0.0000% |
| 2048 | dual_relaxed_visible | 28.80x | 1.111 | 0.0203% | 5.8971% | 0.0000% | 0.0000% |
| 2048 | dual_loose_visible | 35.89x | 0.892 | 0.0398% | 10.6017% | 0.0000% | 0.0000% |
| 2048 | dual_capped | 28.80x | 1.111 | 0.0183% | 3.1643% | 0.7286% | 2.9145% |

## Error Distribution

| Resolution | Variant | Abs err p95 | Abs err p99 | Abs err p99.9 | <=0.5 bias | <=1 bias | <=2 bias | Leak > 1 bias | Shadow > 1 bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 0.0185% | 0.0885% | 0.1385% | 98.5222% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 512 | dual_half_visible | 0.0535% | 0.1415% | 0.1499% | 96.8136% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 512 | dual_visible | 0.0971% | 0.1499% | 0.1499% | 93.3758% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 512 | dual_relaxed_visible | 0.1436% | 0.2080% | 0.2840% | 89.0789% | 97.1283% | 100.0000% | 0.0000% | 2.8717% |
| 512 | dual_loose_visible | 0.2540% | 0.4567% | 0.5976% | 82.1144% | 90.8623% | 96.3268% | 0.0000% | 9.1377% |
| 512 | dual_capped | 0.1280% | 0.2118% | 0.2801% | 90.3851% | 96.2967% | 100.0000% | 3.7033% | 0.0000% |
| 1024 | dual_bias | 0.0415% | 0.1228% | 0.1388% | 97.1322% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 1024 | dual_half_visible | 0.0585% | 0.1248% | 0.1499% | 96.2040% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 1024 | dual_visible | 0.0819% | 0.1345% | 0.1499% | 94.3095% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 1024 | dual_relaxed_visible | 0.1375% | 0.2231% | 0.2785% | 88.8675% | 96.7179% | 100.0000% | 0.0000% | 3.2821% |
| 1024 | dual_loose_visible | 0.2467% | 0.4117% | 0.5442% | 82.6358% | 90.5695% | 96.7587% | 0.0000% | 9.4305% |
| 1024 | dual_capped | 0.1203% | 0.2141% | 0.2709% | 90.0001% | 97.2815% | 100.0000% | 2.7185% | 0.0000% |
| 2048 | dual_bias | 0.0418% | 0.0964% | 0.1347% | 98.1060% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 2048 | dual_half_visible | 0.0620% | 0.1273% | 0.1499% | 96.4242% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 2048 | dual_visible | 0.0818% | 0.1361% | 0.1499% | 94.4060% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 2048 | dual_relaxed_visible | 0.1383% | 0.2187% | 0.2861% | 88.8994% | 96.5703% | 100.0000% | 0.0000% | 3.4297% |
| 2048 | dual_loose_visible | 0.2549% | 0.4034% | 0.5397% | 82.4677% | 90.2084% | 96.4725% | 0.0000% | 9.7916% |
| 2048 | dual_capped | 0.1157% | 0.2080% | 0.2930% | 90.3896% | 97.0855% | 100.0000% | 2.9145% | 0.0000% |

## Depth Texture PCF3 Delta

This estimates the shader-side `Compact SST PCF3` mode against the existing `Depth Texture` PCF3 path.

| Resolution | Variant | SST PCF3 vs Depth PCF3 MAE | SST PCF3 max | SST hard vs Depth PCF3 MAE | SST hard max | @0B hard MAE | @1B hard MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 0.5902% | 77.7778% | 13.3384% | 100.0000% | 10.1309% | 18.2912% |
| 512 | dual_half_visible | 1.1654% | 77.7778% | 13.4065% | 100.0000% | 10.1309% | 18.5636% |
| 512 | dual_visible | 1.7765% | 77.7778% | 13.6111% | 100.0000% | 10.1309% | 18.8372% |
| 512 | dual_relaxed_visible | 2.9317% | 100.0000% | 14.3026% | 100.0000% | 11.4521% | 19.2676% |
| 512 | dual_loose_visible | 5.8398% | 100.0000% | 15.9236% | 100.0000% | 14.7958% | 20.2150% |
| 512 | dual_capped | 2.5442% | 100.0000% | 13.9026% | 100.0000% | 10.1309% | 18.7503% |

## GPU Decompression

This validates the paper-style Compact SST stream by decompressing it on the GPU back to an `r32_float` depth texture.

| Resolution | Variant | GPU == CPU | GPU vs CPU mean | GPU vs CPU max | GPU vs source mean | GPU vs source RMSE | GPU vs source max | Dispatch | Readback |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_visible | True | 0.0000% | 0.0000% | 0.0118% | 0.0336% | 0.1499% | 0.003s | 0.001s |

## Dual Layer Utilization

| Resolution | Variant | Second-hit px | Raw gap mean | Raw gap p95 | Raw gap max | Capped gap mean | Capped gap p95 | Capped gap max | Slack-clamped px |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |
| 512 | dual_half_visible | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |
| 512 | dual_visible | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |
| 512 | dual_relaxed_visible | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |
| 512 | dual_loose_visible | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |
| 512 | dual_capped | 38.3293% | 3.5385% | 14.0934% | 88.9921% | 0.0578% | 0.1500% | 0.1500% | 36.1092% |

## Memory Breakdown

Decomp working set counts the persistent Compact SST stream plus one full-resolution `r32_float` decompressed depth texture.

| Resolution | Variant | Packed bytes | Packed ratio | Packed stream bpt | Tile roots bpt | Packed total bpt | Decomp texture bytes | Packed+decomp bytes | Packed+decomp bpt | Packed+decomp ratio | Fixed64 bytes | Fixed64 ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 215,424 | 4.87x | 6.572 | 0.002 | 6.574 | 1,048,576 | 1,264,000 | 38.574 | 0.83x | 278,656 | 3.76x |
| 512 | dual_half_visible | 171,488 | 6.11x | 5.231 | 0.002 | 5.233 | 1,048,576 | 1,220,064 | 37.233 | 0.86x | 216,768 | 4.84x |
| 512 | dual_visible | 145,056 | 7.23x | 4.425 | 0.002 | 4.427 | 1,048,576 | 1,193,632 | 36.427 | 0.88x | 188,096 | 5.57x |
| 512 | dual_relaxed_visible | 119,208 | 8.80x | 3.636 | 0.002 | 3.638 | 1,048,576 | 1,167,784 | 35.638 | 0.90x | 158,496 | 6.62x |
| 512 | dual_loose_visible | 95,840 | 10.94x | 2.923 | 0.002 | 2.925 | 1,048,576 | 1,144,416 | 34.925 | 0.92x | 132,320 | 7.92x |
| 512 | dual_capped | 119,208 | 8.80x | 3.636 | 0.002 | 3.638 | 1,048,576 | 1,167,784 | 35.638 | 0.90x | 158,496 | 6.62x |
| 1024 | dual_bias | 538,600 | 7.79x | 4.107 | 0.002 | 4.109 | 4,194,304 | 4,732,904 | 36.109 | 0.89x | 721,632 | 5.81x |
| 1024 | dual_half_visible | 394,028 | 10.64x | 3.004 | 0.002 | 3.006 | 4,194,304 | 4,588,332 | 35.006 | 0.91x | 518,688 | 8.09x |
| 1024 | dual_visible | 332,996 | 12.60x | 2.539 | 0.002 | 2.541 | 4,194,304 | 4,527,300 | 34.541 | 0.93x | 443,136 | 9.47x |
| 1024 | dual_relaxed_visible | 264,232 | 15.87x | 2.014 | 0.002 | 2.016 | 4,194,304 | 4,458,536 | 34.016 | 0.94x | 362,464 | 11.57x |
| 1024 | dual_loose_visible | 212,108 | 19.77x | 1.616 | 0.002 | 1.618 | 4,194,304 | 4,406,412 | 33.618 | 0.95x | 299,904 | 13.99x |
| 1024 | dual_capped | 264,212 | 15.87x | 2.014 | 0.002 | 2.016 | 4,194,304 | 4,458,516 | 34.016 | 0.94x | 362,432 | 11.57x |
| 2048 | dual_bias | 1,343,368 | 12.49x | 2.560 | 0.002 | 2.562 | 16,777,216 | 18,120,584 | 34.562 | 0.93x | 1,832,224 | 9.16x |
| 2048 | dual_half_visible | 901,248 | 18.62x | 1.717 | 0.002 | 1.719 | 16,777,216 | 17,678,464 | 33.719 | 0.95x | 1,234,848 | 13.59x |
| 2048 | dual_visible | 740,052 | 22.67x | 1.410 | 0.002 | 1.412 | 16,777,216 | 17,517,268 | 33.412 | 0.96x | 1,025,056 | 16.37x |
| 2048 | dual_relaxed_visible | 582,612 | 28.80x | 1.109 | 0.002 | 1.111 | 16,777,216 | 17,359,828 | 33.111 | 0.97x | 825,568 | 20.32x |
| 2048 | dual_loose_visible | 467,508 | 35.89x | 0.890 | 0.002 | 0.892 | 16,777,216 | 17,244,724 | 32.892 | 0.97x | 677,280 | 24.77x |
| 2048 | dual_capped | 582,584 | 28.80x | 1.109 | 0.002 | 1.111 | 16,777,216 | 17,359,800 | 33.111 | 0.97x | 825,536 | 20.32x |

## Paper Stream Breakdown

| Resolution | Variant | Node words | Branch bpt | 30-bit leaf bpt | 62-bit plane bpt | Tile roots bpt | Packed total bpt |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 53840 | 2.125 | 1.930 | 2.518 | 0.002 | 6.574 |
| 512 | dual_half_visible | 42856 | 1.652 | 1.382 | 2.197 | 0.002 | 5.233 |
| 512 | dual_visible | 36248 | 1.434 | 1.313 | 1.678 | 0.002 | 4.427 |
| 512 | dual_relaxed_visible | 29786 | 1.208 | 1.199 | 1.229 | 0.002 | 3.638 |
| 512 | dual_loose_visible | 23944 | 1.008 | 1.113 | 0.802 | 0.002 | 2.925 |
| 512 | dual_capped | 29786 | 1.208 | 1.199 | 1.229 | 0.002 | 3.638 |
| 1024 | dual_bias | 134586 | 1.375 | 1.396 | 1.336 | 0.002 | 4.109 |
| 1024 | dual_half_visible | 98443 | 0.988 | 0.951 | 1.065 | 0.002 | 3.006 |
| 1024 | dual_visible | 83185 | 0.844 | 0.840 | 0.855 | 0.002 | 2.541 |
| 1024 | dual_relaxed_visible | 65994 | 0.690 | 0.749 | 0.575 | 0.002 | 2.016 |
| 1024 | dual_loose_visible | 52963 | 0.571 | 0.670 | 0.376 | 0.002 | 1.618 |
| 1024 | dual_capped | 65989 | 0.690 | 0.749 | 0.575 | 0.002 | 2.016 |
| 2048 | dual_bias | 335586 | 0.872 | 0.932 | 0.756 | 0.002 | 2.562 |
| 2048 | dual_half_visible | 225056 | 0.587 | 0.636 | 0.493 | 0.002 | 1.719 |
| 2048 | dual_visible | 184757 | 0.487 | 0.544 | 0.379 | 0.002 | 1.412 |
| 2048 | dual_relaxed_visible | 145397 | 0.392 | 0.463 | 0.254 | 0.002 | 1.111 |
| 2048 | dual_loose_visible | 116621 | 0.321 | 0.400 | 0.168 | 0.002 | 0.892 |
| 2048 | dual_capped | 145390 | 0.392 | 0.463 | 0.254 | 0.002 | 1.111 |

## Branch Offset Packing

| Resolution | Variant | 10-bit start level | 13-bit branches | Max 13-bit offset | 13-bit capacity | 10-bit branches | Max 10-bit offset | 10-bit capacity | Overflows |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 7 | 8702 | 4665 | 56.9528% | 0 | 0 | 0.0000% | 0 |
| 512 | dual_half_visible | 7 | 6768 | 3878 | 47.3446% | 0 | 0 | 0.0000% | 0 |
| 512 | dual_visible | 7 | 5872 | 3057 | 37.3215% | 0 | 0 | 0.0000% | 0 |
| 512 | dual_relaxed_visible | 7 | 4947 | 1901 | 23.2084% | 0 | 0 | 0.0000% | 0 |
| 512 | dual_loose_visible | 7 | 4129 | 1313 | 16.0298% | 0 | 0 | 0.0000% | 0 |
| 512 | dual_capped | 7 | 4947 | 1901 | 23.2084% | 0 | 0 | 0.0000% | 0 |
| 1024 | dual_bias | 7 | 22527 | 5753 | 70.2356% | 0 | 0 | 0.0000% | 0 |
| 1024 | dual_half_visible | 7 | 16185 | 3580 | 43.7065% | 0 | 0 | 0.0000% | 0 |
| 1024 | dual_visible | 7 | 13824 | 2541 | 31.0219% | 0 | 0 | 0.0000% | 0 |
| 1024 | dual_relaxed_visible | 7 | 11303 | 2196 | 26.8099% | 0 | 0 | 0.0000% | 0 |
| 1024 | dual_loose_visible | 7 | 9348 | 1866 | 22.7811% | 0 | 0 | 0.0000% | 0 |
| 1024 | dual_capped | 7 | 11302 | 2196 | 26.8099% | 0 | 0 | 0.0000% | 0 |
| 2048 | dual_bias | 7 | 57161 | 5688 | 69.4421% | 0 | 0 | 0.0000% | 0 |
| 2048 | dual_half_visible | 7 | 38493 | 4751 | 58.0027% | 0 | 0 | 0.0000% | 0 |
| 2048 | dual_visible | 7 | 31937 | 4416 | 53.9128% | 0 | 0 | 0.0000% | 0 |
| 2048 | dual_relaxed_visible | 7 | 25703 | 3960 | 48.3457% | 0 | 0 | 0.0000% | 0 |
| 2048 | dual_loose_visible | 7 | 21069 | 3229 | 39.4213% | 0 | 0 | 0.0000% | 0 |
| 2048 | dual_capped | 7 | 25702 | 3960 | 48.3457% | 0 | 0 | 0.0000% | 0 |

## Forced Leaf Diagnostics

| Resolution | Variant | Forced leaves | Forced pixels | Forced mean err | Forced max err |
|---:|---|---:|---:|---:|---:|
| 512 | dual_bias | 0 | 0.0000% | 0.0000% | 0.0000% |
| 512 | dual_half_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 512 | dual_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 512 | dual_relaxed_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 512 | dual_loose_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 512 | dual_capped | 0 | 0.0000% | 0.0000% | 0.0000% |
| 1024 | dual_bias | 0 | 0.0000% | 0.0000% | 0.0000% |
| 1024 | dual_half_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 1024 | dual_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 1024 | dual_relaxed_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 1024 | dual_loose_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 1024 | dual_capped | 0 | 0.0000% | 0.0000% | 0.0000% |
| 2048 | dual_bias | 0 | 0.0000% | 0.0000% | 0.0000% |
| 2048 | dual_half_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 2048 | dual_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 2048 | dual_relaxed_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 2048 | dual_loose_visible | 0 | 0.0000% | 0.0000% | 0.0000% |
| 2048 | dual_capped | 0 | 0.0000% | 0.0000% | 0.0000% |

## Node Type Composition

| Resolution | Variant | Branches | Branch % | 30-bit leaves | 30-bit % | 62-bit planes | 62-bit % | Total nodes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 8702 | 24.9885% | 15808 | 45.3940% | 10314 | 29.6175% | 34824 |
| 512 | dual_half_visible | 6768 | 24.9852% | 11320 | 41.7897% | 9000 | 33.2250% | 27088 |
| 512 | dual_visible | 5872 | 24.9830% | 10760 | 45.7794% | 6872 | 29.2376% | 23504 |
| 512 | dual_relaxed_visible | 4947 | 24.9798% | 9822 | 49.5960% | 5035 | 25.4242% | 19804 |
| 512 | dual_loose_visible | 4129 | 24.9758% | 9120 | 55.1657% | 3283 | 19.8585% | 16532 |
| 512 | dual_capped | 4947 | 24.9798% | 9822 | 49.5960% | 5035 | 25.4242% | 19804 |
| 1024 | dual_bias | 22527 | 24.9823% | 45758 | 50.7452% | 21887 | 24.2725% | 90172 |
| 1024 | dual_half_visible | 16185 | 24.9753% | 31165 | 48.0912% | 17454 | 26.9335% | 64804 |
| 1024 | dual_visible | 13824 | 24.9711% | 27535 | 49.7381% | 14001 | 25.2908% | 55360 |
| 1024 | dual_relaxed_visible | 11303 | 24.9647% | 24558 | 54.2407% | 9415 | 20.7947% | 45276 |
| 1024 | dual_loose_visible | 9348 | 24.9573% | 21949 | 58.5994% | 6159 | 16.4433% | 37456 |
| 1024 | dual_capped | 11302 | 24.9647% | 24555 | 54.2388% | 9415 | 20.7965% | 45272 |
| 2048 | dual_bias | 57161 | 24.9720% | 122214 | 53.3919% | 49525 | 21.6361% | 228900 |
| 2048 | dual_half_visible | 38493 | 24.9585% | 83400 | 54.0758% | 32335 | 20.9657% | 154228 |
| 2048 | dual_visible | 31937 | 24.9500% | 71251 | 55.6631% | 24816 | 19.3869% | 128004 |
| 2048 | dual_relaxed_visible | 25703 | 24.9379% | 60739 | 58.9310% | 16626 | 16.1311% | 103068 |
| 2048 | dual_loose_visible | 21069 | 24.9243% | 52443 | 62.0392% | 11020 | 13.0365% | 84532 |
| 2048 | dual_capped | 25702 | 24.9379% | 60738 | 58.9323% | 16624 | 16.1298% | 103064 |

## Visibility Probe Breakdown

| Resolution | Variant | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | dual_bias | 0.0036% | 0.0000% | 0.0000% | 0.0145% | 0.0000% | 0.0000% | 0.0036% |
| 512 | dual_half_visible | 0.8914% | 0.0000% | 0.0000% | 3.5656% | 0.0000% | 0.0000% | 0.8914% |
| 512 | dual_visible | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| 512 | dual_relaxed_visible | 5.1992% | 2.8713% | 6.4827% | 11.4429% | 0.0000% | 0.0000% | 5.1992% |
| 512 | dual_loose_visible | 10.0773% | 9.1370% | 13.3266% | 17.8455% | 0.0000% | 0.0000% | 10.0773% |
| 512 | dual_capped | 3.3482% | 0.0000% | 1.9817% | 7.7076% | 3.7033% | 0.9258% | 2.4223% |
| 1024 | dual_bias | 0.0065% | 0.0000% | 0.0000% | 0.0258% | 0.0000% | 0.0000% | 0.0065% |
| 1024 | dual_half_visible | 0.9603% | 0.0000% | 0.0000% | 3.8413% | 0.0000% | 0.0000% | 0.9603% |
| 1024 | dual_visible | 2.1022% | 0.0000% | 1.6140% | 6.7947% | 0.0000% | 0.0000% | 2.1022% |
| 1024 | dual_relaxed_visible | 5.5929% | 3.2820% | 6.8816% | 12.2081% | 0.0000% | 0.0000% | 5.5929% |
| 1024 | dual_loose_visible | 10.1970% | 9.4304% | 13.3316% | 18.0260% | 0.0000% | 0.0000% | 10.1970% |
| 1024 | dual_capped | 3.1276% | 0.0000% | 2.1680% | 7.6237% | 2.7187% | 0.6797% | 2.4479% |
| 2048 | dual_bias | 0.0122% | 0.0000% | 0.0000% | 0.0488% | 0.0000% | 0.0000% | 0.0122% |
| 2048 | dual_half_visible | 1.1391% | 0.0000% | 0.0000% | 4.5566% | 0.0000% | 0.0000% | 1.1391% |
| 2048 | dual_visible | 2.3436% | 0.0000% | 1.6765% | 7.6977% | 0.0000% | 0.0000% | 2.3436% |
| 2048 | dual_relaxed_visible | 5.8971% | 3.4296% | 7.0563% | 13.1024% | 0.0000% | 0.0000% | 5.8971% |
| 2048 | dual_loose_visible | 10.6017% | 9.7915% | 13.7105% | 18.9047% | 0.0000% | 0.0000% | 10.6017% |
| 2048 | dual_capped | 3.1643% | 0.0000% | 2.0268% | 7.7158% | 2.9145% | 0.7286% | 2.4356% |

## PCF-Aware Sweep Candidates

| Source | Resolution | Variant | Min leaf | Packed | Packed bpt | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias | <=1 bias |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_loose_visible | 1 | 10.94x | 2.925 | 0.92x | 5.8398% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_loose_visible | 2 | 13.16x | 2.431 | 0.93x | 6.1036% | 10.6684% | 0.0000% | 9.9091% | 90.0909% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_loose_visible | 4 | 24.22x | 1.321 | 0.96x | 9.2417% | 14.9347% | 0.0006% | 16.3994% | 83.5983% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_loose_visible | 8 | 56.64x | 0.565 | 0.98x | 14.2423% | 20.6970% | 0.0025% | 25.2102% | 74.7799% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_relaxed_visible | 1 | 8.80x | 3.638 | 0.90x | 2.9317% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_relaxed_visible | 2 | 10.52x | 3.042 | 0.91x | 3.2294% | 5.9100% | 0.0000% | 3.8025% | 96.1975% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_relaxed_visible | 4 | 19.97x | 1.602 | 0.95x | 6.7984% | 11.2012% | 0.0006% | 11.7050% | 88.2927% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_relaxed_visible | 8 | 54.23x | 0.590 | 0.98x | 13.7349% | 19.9996% | 0.0025% | 24.2527% | 75.7374% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_visible | 1 | 7.23x | 4.427 | 0.88x | 1.7765% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_visible | 2 | 8.68x | 3.688 | 0.90x | 2.1255% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_visible | 4 | 18.52x | 1.728 | 0.95x | 6.1190% | 9.8669% | 0.0006% | 10.3977% | 89.6000% |
| sst_bistro_512_pcf_minleaf_sweep.json | 512 | dual_visible | 8 | 52.90x | 0.605 | 0.98x | 13.4905% | 19.5634% | 0.0025% | 23.7095% | 76.2806% |
| sst_bistro_512_pcf_profile_sweep.json | 512 | dual_bias | 1 | 4.87x | 6.574 | 0.83x | 0.5902% | 0.0036% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | 512 | dual_half_visible | 1 | 6.11x | 5.233 | 0.86x | 1.1654% | 0.8914% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | 512 | dual_loose_visible | 1 | 10.94x | 2.925 | 0.92x | 5.8398% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| sst_bistro_512_pcf_profile_sweep.json | 512 | dual_relaxed_visible | 1 | 8.80x | 3.638 | 0.90x | 2.9317% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_pcf_profile_sweep.json | 512 | dual_visible | 1 | 7.23x | 4.427 | 0.88x | 1.7765% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | 512 | single | 1 | 5.77x | 5.546 | 0.85x | 2.0754% | 9.4118% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | 1024 | dual_relaxed_visible | 1 | 15.87x | 2.016 | 0.94x | 4.2512% | 5.5929% | 0.0000% | 3.2821% | 96.7179% |
| sst_bistro_1024_pcf_minleaf_sweep.json | 1024 | dual_relaxed_visible | 2 | 19.06x | 1.679 | 0.95x | 4.4288% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| sst_bistro_1024_pcf_minleaf_sweep.json | 1024 | dual_visible | 1 | 12.60x | 2.541 | 0.93x | 2.1657% | 2.1022% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | 1024 | dual_visible | 2 | 15.05x | 2.127 | 0.94x | 2.3723% | 2.5915% | 0.0000% | 0.6459% | 99.3541% |
| sst_bistro_2048_pcf_minleaf_sweep.json | 2048 | dual_relaxed_visible | 1 | 28.80x | 1.111 | 0.97x | 5.0182% | 5.8971% | 0.0000% | 3.4297% | 96.5703% |
| sst_bistro_2048_pcf_minleaf_sweep.json | 2048 | dual_relaxed_visible | 2 | 34.92x | 0.916 | 0.97x | 5.1311% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| sst_bistro_2048_pcf_minleaf_sweep.json | 2048 | dual_visible | 1 | 22.67x | 1.412 | 0.96x | 2.3787% | 2.3436% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_2048_pcf_minleaf_sweep.json | 2048 | dual_visible | 2 | 27.31x | 1.172 | 0.96x | 2.5124% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |

## Sweep Recommendations

| Source | Constraint | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Packed bpt | Packed+decomp ratio | Vis mismatch | PCF3 MAE | False lit | False shadow | Leak > 1 bias | Shadow > 1 bias | <=1 bias |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sst_bistro_512_forcecap_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_forcecap_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_forcecap_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_forcecap_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_forcecap_sweep.json | max_compression_mean_error_le_0_1_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_forcecap_sweep.json | max_compression_false_lit_le_1_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_minleaf_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_minleaf_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_minleaf_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_minleaf_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_minleaf_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_minleaf_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_minleaf_sweep.json | max_compression_mean_error_le_0_1_percent | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | n/a | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_minleaf_sweep.json | max_compression_false_lit_le_1_percent | dual_visible | 128 | 8 | none | no | 0.001000 | 0.0015 | 0 | 52.90x | 0.605 | 0.98x | 19.5634% | n/a | 0.0025% | 19.5609% | 0.0099% | 23.7095% | 76.2806% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | 2.1255% | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 8.68x | 3.688 | 0.90x | 3.1948% | 2.1255% | 0.0000% | 3.1948% | 0.0000% | 1.1528% | 98.8472% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 3.042 | 0.91x | 5.9100% | 3.2294% | 0.0000% | 5.9100% | 0.0000% | 3.8025% | 96.1975% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 3.042 | 0.91x | 5.9100% | 3.2294% | 0.0000% | 5.9100% | 0.0000% | 3.8025% | 96.1975% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_mean_error_le_0_1_percent | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 2.925 | 0.92x | 10.0773% | 5.8398% | 0.0000% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_false_lit_le_1_percent | dual_loose_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 56.64x | 0.565 | 0.98x | 20.6970% | 14.2423% | 0.0025% | 20.6945% | 0.0099% | 25.2102% | 74.7799% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_pcf3_mae_le_2_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_pcf3_mae_le_3_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | 2.9317% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_pcf_minleaf_sweep.json | max_compression_pcf3_mae_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 3.042 | 0.91x | 5.9100% | 3.2294% | 0.0000% | 5.9100% | 0.0000% | 3.8025% | 96.1975% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_vis_mismatch_le_1_percent | dual_half_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 6.11x | 5.233 | 0.86x | 0.8914% | 1.1654% | 0.0000% | 0.8914% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | 2.9317% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | 2.9317% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_mean_error_le_0_1_percent | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 2.925 | 0.92x | 10.0773% | 5.8398% | 0.0000% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_false_lit_le_1_percent | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 2.925 | 0.92x | 10.0773% | 5.8398% | 0.0000% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_pcf3_mae_le_1_percent | dual_bias | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 4.87x | 6.574 | 0.83x | 0.0036% | 0.5902% | 0.0000% | 0.0036% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_pcf3_mae_le_2_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | 1.7765% | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_pcf3_mae_le_3_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | 2.9317% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_pcf_profile_sweep.json | max_compression_pcf3_mae_le_5_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | 2.9317% | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_profile_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_profile_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_profile_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_profile_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_profile_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | n/a | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_profile_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | n/a | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_profile_sweep.json | max_compression_mean_error_le_0_1_percent | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 2.925 | 0.92x | 10.0773% | n/a | 0.0000% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| sst_bistro_512_profile_sweep.json | max_compression_false_lit_le_1_percent | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 2.925 | 0.92x | 10.0773% | n/a | 0.0000% | 10.0773% | 0.0000% | 9.1377% | 90.8623% |
| sst_bistro_512_qradius_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_qradius_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_qradius_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_qradius_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_qradius_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_qradius_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_qradius_sweep.json | max_compression_mean_error_le_0_1_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_qradius_sweep.json | max_compression_false_lit_le_1_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 4.411 | 0.88x | 2.3393% | n/a | 0.0000% | 2.3393% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_slack_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_slack_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_slack_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_slack_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_slack_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | n/a | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_slack_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | n/a | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_slack_sweep.json | max_compression_mean_error_le_0_1_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | n/a | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_slack_sweep.json | max_compression_false_lit_le_1_percent | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 5.1992% | n/a | 0.0000% | 5.1992% | 0.0000% | 2.8717% | 97.1283% |
| sst_bistro_512_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_vis_mismatch_le_1_percent | dual_half_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 6.11x | 5.233 | 0.86x | 0.8914% | n/a | 0.0000% | 0.8914% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_mean_error_le_0_1_percent | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.427 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_sweep.json | max_compression_false_lit_le_1_percent | dual_capped | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 3.638 | 0.90x | 3.3482% | n/a | 0.9258% | 2.4223% | 3.7033% | 0.0000% | 96.2967% |
| sst_bistro_512_tile_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_tile_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_tile_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_tile_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_tile_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_tile_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_tile_sweep.json | max_compression_mean_error_le_0_1_percent | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_512_tile_sweep.json | max_compression_false_lit_le_1_percent | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 4.426 | 0.88x | 2.3163% | n/a | 0.0000% | 2.3163% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 12.60x | 2.541 | 0.93x | 2.1022% | 2.1657% | 0.0000% | 2.1022% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 12.60x | 2.541 | 0.93x | 2.1022% | 2.1657% | 0.0000% | 2.1022% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 2.127 | 0.94x | 2.5915% | 2.3723% | 0.0000% | 2.5915% | 0.0000% | 0.6459% | 99.3541% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 2.127 | 0.94x | 2.5915% | 2.3723% | 0.0000% | 2.5915% | 0.0000% | 0.6459% | 99.3541% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 1.679 | 0.95x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 1.679 | 0.95x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_mean_error_le_0_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 1.679 | 0.95x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_false_lit_le_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 1.679 | 0.95x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_pcf3_mae_le_3_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 2.127 | 0.94x | 2.5915% | 2.3723% | 0.0000% | 2.5915% | 0.0000% | 0.6459% | 99.3541% |
| sst_bistro_1024_pcf_minleaf_sweep.json | max_compression_pcf3_mae_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 1.679 | 0.95x | 5.9924% | 4.4288% | 0.0000% | 5.9924% | 0.0000% | 3.8087% | 96.1913% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_no_false_lit_no_gt_bias_error | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 22.67x | 1.412 | 0.96x | 2.3436% | 2.3787% | 0.0000% | 2.3436% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_vis_mismatch_le_2_5_percent | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 22.67x | 1.412 | 0.96x | 2.3436% | 2.3787% | 0.0000% | 2.3436% | 0.0000% | 0.0000% | 100.0000% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_vis_mismatch_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_false_shadow_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_shadow_over_1_bias_le_5_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_abs_error_within_1_bias_ge_95_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_mean_error_le_0_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_false_lit_le_1_percent | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.916 | 0.97x | 6.1282% | 5.1311% | 0.0000% | 6.1282% | 0.0000% | 3.7343% | 96.2657% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_pcf3_mae_le_3_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |
| sst_bistro_2048_pcf_minleaf_sweep.json | max_compression_pcf3_mae_le_5_percent | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 1.172 | 0.96x | 2.6270% | 2.5124% | 0.0000% | 2.6270% | 0.0000% | 0.3737% | 99.6263% |

## Pareto Fronts

| Source | Variant | Tile | Min leaf | Force cap | Bias split | Plane err | Slack | Q radius | Packed | Mean depth err | Forced px | Vis mismatch | False lit | False shadow | Leak > 1 bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sst_bistro_512_forcecap_sweep.json | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 0.0907% | 2.3087% | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| sst_bistro_512_forcecap_sweep.json | dual_visible | 128 | 2 | 0.010000 | no | 0.001000 | 0.0015 | 0 | 7.77x | 0.5754% | 0.9567% | 2.9521% | 0.2768% | 2.6753% | 1.1070% |
| sst_bistro_512_forcecap_sweep.json | dual_visible | 128 | 2 | 0.005000 | no | 0.001000 | 0.0015 | 0 | 7.64x | 0.6828% | 0.7431% | 2.9251% | 0.3300% | 2.5951% | 1.3199% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0.001000 | 0.0015 | 0 | 8.68x | 0.0907% | 2.3087% | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 4 | none | no | 0.001000 | 0.0015 | 0 | 18.52x | 0.8184% | 15.3687% | 9.8669% | 0.0006% | 9.8663% | 0.0023% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 8 | none | no | 0.001000 | 0.0015 | 0 | 52.90x | 2.1083% | 29.1992% | 19.5634% | 0.0025% | 19.5609% | 0.0099% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 8.68x | 0.0907% | 2.3087% | 3.1948% | 0.0000% | 3.1948% | 0.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0194% | 0.0000% | 5.1992% | 0.0000% | 5.1992% | 0.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 10.52x | 0.0976% | 1.8616% | 5.9100% | 0.0000% | 5.9100% | 0.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 0.0409% | 0.0000% | 10.0773% | 0.0000% | 10.0773% | 0.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 13.16x | 0.1184% | 1.5427% | 10.6684% | 0.0000% | 10.6684% | 0.0000% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 4 | none | no | 0.001500 | 0.0015 | 0 | 18.52x | 0.8184% | 15.3687% | 9.8669% | 0.0006% | 9.8663% | 0.0023% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 4 | none | no | 0.001500 | 0.0015 | 0 | 19.97x | 0.8219% | 12.1887% | 11.2012% | 0.0006% | 11.2006% | 0.0023% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 4 | none | no | 0.001500 | 0.0015 | 0 | 24.22x | 0.8383% | 9.7717% | 14.9347% | 0.0006% | 14.9342% | 0.0023% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 52.90x | 2.1083% | 29.1992% | 19.5634% | 0.0025% | 19.5609% | 0.0099% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 54.23x | 2.1095% | 26.8555% | 19.9996% | 0.0025% | 19.9971% | 0.0099% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 8 | none | no | 0.001500 | 0.0015 | 0 | 56.64x | 2.1127% | 22.2412% | 20.6970% | 0.0025% | 20.6945% | 0.0099% |
| sst_bistro_512_pcf_profile_sweep.json | dual_bias | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 4.87x | 0.0036% | 0.0000% | 0.0036% | 0.0000% | 0.0036% | 0.0000% |
| sst_bistro_512_pcf_profile_sweep.json | dual_half_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 6.11x | 0.0073% | 0.0000% | 0.8914% | 0.0000% | 0.8914% | 0.0000% |
| sst_bistro_512_pcf_profile_sweep.json | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_512_pcf_profile_sweep.json | single | 128 | 1 | none | no | 0.001500 | None | 0 | 5.77x | 0.0052% | 0.0000% | 9.4118% | 0.0000% | 9.4118% | 0.0000% |
| sst_bistro_512_pcf_profile_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0194% | 0.0000% | 5.1992% | 0.0000% | 5.1992% | 0.0000% |
| sst_bistro_512_pcf_profile_sweep.json | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 0.0409% | 0.0000% | 10.0773% | 0.0000% | 10.0773% | 0.0000% |
| sst_bistro_512_profile_sweep.json | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_512_profile_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0194% | 0.0000% | 5.1992% | 0.0000% | 5.1992% | 0.0000% |
| sst_bistro_512_profile_sweep.json | dual_loose_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 10.94x | 0.0409% | 0.0000% | 10.0773% | 0.0000% | 10.0773% | 0.0000% |
| sst_bistro_512_profile_sweep.json | dual_capped | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0181% | 0.0000% | 3.3482% | 0.9258% | 2.4223% | 3.7033% |
| sst_bistro_512_qradius_sweep.json | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_512_qradius_sweep.json | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 1 | 7.26x | 0.0119% | 0.0000% | 2.3393% | 0.0000% | 2.3393% | 0.0000% |
| sst_bistro_512_slack_sweep.json | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_512_slack_sweep.json | dual_visible | 128 | 1 | none | no | 0.001500 | 0.00075 | 0 | 6.68x | 0.0080% | 0.0000% | 2.4752% | 0.0000% | 2.4752% | 0.0000% |
| sst_bistro_512_slack_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0194% | 0.0000% | 5.1992% | 0.0000% | 5.1992% | 0.0000% |
| sst_bistro_512_slack_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.00075 | 0 | 8.50x | 0.0180% | 0.0000% | 6.1652% | 0.0000% | 6.1652% | 0.0000% |
| sst_bistro_512_sweep.json | dual_bias | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 4.87x | 0.0036% | 0.0000% | 0.0036% | 0.0000% | 0.0036% | 0.0000% |
| sst_bistro_512_sweep.json | dual_half_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 6.11x | 0.0073% | 0.0000% | 0.8914% | 0.0000% | 0.8914% | 0.0000% |
| sst_bistro_512_sweep.json | dual_visible | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_512_sweep.json | dual_capped | 128 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.85x | 0.0144% | 0.0000% | 2.4583% | 0.7122% | 1.7461% | 2.8488% |
| sst_bistro_512_sweep.json | dual_capped | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 8.80x | 0.0181% | 0.0000% | 3.3482% | 0.9258% | 2.4223% | 3.7033% |
| sst_bistro_512_sweep.json | dual_capped | 128 | 1 | none | no | 0.001000 | 0.003 | 0 | 8.23x | 0.0193% | 0.0000% | 2.5042% | 1.1323% | 1.3719% | 4.5288% |
| sst_bistro_512_sweep.json | dual_capped | 128 | 1 | none | no | 0.002500 | 0.0015 | 0 | 10.33x | 0.0262% | 0.0000% | 5.1989% | 1.3771% | 3.8218% | 5.5084% |
| sst_bistro_512_sweep.json | dual_capped | 128 | 1 | none | no | 0.001500 | 0.003 | 0 | 9.18x | 0.0241% | 0.0000% | 3.3706% | 1.3948% | 1.9757% | 5.5790% |
| sst_bistro_512_sweep.json | dual_capped | 128 | 1 | none | no | 0.002500 | 0.003 | 0 | 10.74x | 0.0370% | 0.0000% | 5.6034% | 2.0377% | 3.5657% | 8.1509% |
| sst_bistro_512_tile_sweep.json | dual_visible | 256 | 1 | none | no | 0.001000 | 0.0015 | 0 | 7.23x | 0.0118% | 0.0000% | 2.3163% | 0.0000% | 2.3163% | 0.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 12.60x | 0.0109% | 0.0000% | 2.1022% | 0.0000% | 2.1022% | 0.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 15.05x | 0.0509% | 1.2932% | 2.5915% | 0.0000% | 2.5915% | 0.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 15.87x | 0.0200% | 0.0000% | 5.5929% | 0.0000% | 5.5929% | 0.0000% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 19.06x | 0.0597% | 1.0532% | 5.9924% | 0.0000% | 5.9924% | 0.0000% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 22.67x | 0.0112% | 0.0000% | 2.3436% | 0.0000% | 2.3436% | 0.0000% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 27.31x | 0.0322% | 0.7489% | 2.6270% | 0.0000% | 2.6270% | 0.0000% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0.001500 | 0.0015 | 0 | 28.80x | 0.0203% | 0.0000% | 5.8971% | 0.0000% | 5.8971% | 0.0000% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 2 | none | no | 0.001500 | 0.0015 | 0 | 34.92x | 0.0412% | 0.6092% | 6.1282% | 0.0000% | 6.1282% | 0.0000% |

## Pareto Visibility Probes

| Source | Variant | Tile | Min leaf | Force cap | Bias split | Q radius | Aggregate | @0B | @0.5B | @1B | @2B | False lit | False shadow |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sst_bistro_512_forcecap_sweep.json | dual_visible | 128 | 2 | none | no | 0 | 3.1948% | 1.1528% | 3.4721% | 8.1543% | 0.0000% | 0.0000% | 3.1948% |
| sst_bistro_512_forcecap_sweep.json | dual_visible | 128 | 2 | 0.010000 | no | 0 | 2.9521% | 0.4768% | 2.7962% | 7.4284% | 1.1070% | 0.2768% | 2.6753% |
| sst_bistro_512_forcecap_sweep.json | dual_visible | 128 | 2 | 0.005000 | no | 0 | 2.9251% | 0.3700% | 2.6894% | 7.3212% | 1.3199% | 0.3300% | 2.5951% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0 | 3.1948% | 1.1528% | 3.4721% | 8.1543% | 0.0000% | 0.0000% | 3.1948% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 4 | none | no | 0 | 9.8669% | 10.3977% | 12.5111% | 16.5565% | 0.0023% | 0.0006% | 9.8663% |
| sst_bistro_512_minleaf_sweep.json | dual_visible | 128 | 8 | none | no | 0 | 19.5634% | 23.7091% | 25.8389% | 28.6957% | 0.0099% | 0.0025% | 19.5609% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0 | 3.1948% | 1.1528% | 3.4721% | 8.1543% | 0.0000% | 0.0000% | 3.1948% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.1992% | 2.8713% | 6.4827% | 11.4429% | 0.0000% | 0.0000% | 5.1992% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 2 | none | no | 0 | 5.9100% | 3.8021% | 7.4135% | 12.4245% | 0.0000% | 0.0000% | 5.9100% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 1 | none | no | 0 | 10.0773% | 9.1370% | 13.3266% | 17.8455% | 0.0000% | 0.0000% | 10.0773% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 2 | none | no | 0 | 10.6684% | 9.9083% | 14.0980% | 18.6672% | 0.0000% | 0.0000% | 10.6684% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 4 | none | no | 0 | 9.8669% | 10.3977% | 12.5111% | 16.5565% | 0.0023% | 0.0006% | 9.8663% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 4 | none | no | 0 | 11.2012% | 11.7046% | 14.4775% | 18.6203% | 0.0023% | 0.0006% | 11.2006% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 4 | none | no | 0 | 14.9347% | 16.3986% | 19.7819% | 23.5561% | 0.0023% | 0.0006% | 14.9342% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_visible | 128 | 8 | none | no | 0 | 19.5634% | 23.7091% | 25.8389% | 28.6957% | 0.0099% | 0.0025% | 19.5609% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 8 | none | no | 0 | 19.9996% | 24.2519% | 26.4523% | 29.2843% | 0.0099% | 0.0025% | 19.9971% |
| sst_bistro_512_pcf_minleaf_sweep.json | dual_loose_visible | 128 | 8 | none | no | 0 | 20.6970% | 25.2090% | 27.4078% | 30.1613% | 0.0099% | 0.0025% | 20.6945% |
| sst_bistro_512_pcf_profile_sweep.json | dual_bias | 128 | 1 | none | no | 0 | 0.0036% | 0.0000% | 0.0000% | 0.0145% | 0.0000% | 0.0000% | 0.0036% |
| sst_bistro_512_pcf_profile_sweep.json | dual_half_visible | 128 | 1 | none | no | 0 | 0.8914% | 0.0000% | 0.0000% | 3.5656% | 0.0000% | 0.0000% | 0.8914% |
| sst_bistro_512_pcf_profile_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| sst_bistro_512_pcf_profile_sweep.json | single | 128 | 1 | none | no | 0 | 9.4118% | 0.0000% | 2.3865% | 35.2608% | 0.0000% | 0.0000% | 9.4118% |
| sst_bistro_512_pcf_profile_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.1992% | 2.8713% | 6.4827% | 11.4429% | 0.0000% | 0.0000% | 5.1992% |
| sst_bistro_512_pcf_profile_sweep.json | dual_loose_visible | 128 | 1 | none | no | 0 | 10.0773% | 9.1370% | 13.3266% | 17.8455% | 0.0000% | 0.0000% | 10.0773% |
| sst_bistro_512_profile_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| sst_bistro_512_profile_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.1992% | 2.8713% | 6.4827% | 11.4429% | 0.0000% | 0.0000% | 5.1992% |
| sst_bistro_512_profile_sweep.json | dual_loose_visible | 128 | 1 | none | no | 0 | 10.0773% | 9.1370% | 13.3266% | 17.8455% | 0.0000% | 0.0000% | 10.0773% |
| sst_bistro_512_profile_sweep.json | dual_capped | 128 | 1 | none | no | 0 | 3.3482% | 0.0000% | 1.9817% | 7.7076% | 3.7033% | 0.9258% | 2.4223% |
| sst_bistro_512_slack_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.3163% | 0.0000% | 2.3178% | 6.9473% | 0.0000% | 0.0000% | 2.3163% |
| sst_bistro_512_slack_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.4752% | 0.0000% | 2.5265% | 7.3742% | 0.0000% | 0.0000% | 2.4752% |
| sst_bistro_512_slack_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.1992% | 2.8713% | 6.4827% | 11.4429% | 0.0000% | 0.0000% | 5.1992% |
| sst_bistro_512_slack_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0 | 6.1652% | 3.9757% | 7.7187% | 12.9665% | 0.0000% | 0.0000% | 6.1652% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.1022% | 0.0000% | 1.6140% | 6.7947% | 0.0000% | 0.0000% | 2.1022% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0 | 2.5915% | 0.6459% | 2.2606% | 7.4596% | 0.0000% | 0.0000% | 2.5915% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.5929% | 3.2820% | 6.8816% | 12.2081% | 0.0000% | 0.0000% | 5.5929% |
| sst_bistro_1024_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 2 | none | no | 0 | 5.9924% | 3.8086% | 7.4082% | 12.7526% | 0.0000% | 0.0000% | 5.9924% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_visible | 128 | 1 | none | no | 0 | 2.3436% | 0.0000% | 1.6765% | 7.6977% | 0.0000% | 0.0000% | 2.3436% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_visible | 128 | 2 | none | no | 0 | 2.6270% | 0.3737% | 2.0510% | 8.0831% | 0.0000% | 0.0000% | 2.6270% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 1 | none | no | 0 | 5.8971% | 3.4296% | 7.0563% | 13.1024% | 0.0000% | 0.0000% | 5.8971% |
| sst_bistro_2048_pcf_minleaf_sweep.json | dual_relaxed_visible | 128 | 2 | none | no | 0 | 6.1282% | 3.7342% | 7.3609% | 13.4178% | 0.0000% | 0.0000% | 6.1282% |
