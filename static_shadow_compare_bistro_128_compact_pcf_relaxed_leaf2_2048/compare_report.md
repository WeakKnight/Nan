# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `128x128`
- Frames: `1`
- Shadow resolution: `2048`
- Reference mode: `depth`
- Bake time: `0.004s`
- Encode time: `61.243s`
- SST encoded: `True`
- SST profile: `Dual Relaxed Visible`
- SST tile/leaf: `128/2`
- SST packed ratio: `34.92x`
- SST packed bpt: `0.916`
- SST packed+decomp ratio: `0.97x`

## SST Encoding

| Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 77516 | 480,404 | 0.916 | 34.92x | 17,257,620 | 0.97x | 5.1311% | 6.1282% | 0.0000% | 3.7343% |

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\00_depth.png` | 0.036 |
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\01_compact-pcf.png` | 0.014 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\diff_depth_vs_compact-pcf.png` | 0.0372% | 0.7482% | 36.0784% | 0.5005% | 42.52 dB | 0.2192 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\diff_depth_vs_compact-pcf.png` | 0.0372% | 0.7482% | 36.0784% | 0.5005% | 42.52 dB |
