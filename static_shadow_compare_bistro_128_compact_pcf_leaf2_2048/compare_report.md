# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `128x128`
- Frames: `1`
- Shadow resolution: `2048`
- Reference mode: `depth`
- Bake time: `0.005s`
- Encode time: `60.990s`
- SST encoded: `True`
- SST profile: `Dual Visible`
- SST tile/leaf: `128/2`
- SST packed ratio: `27.31x`
- SST packed bpt: `1.172`
- SST packed+decomp ratio: `0.96x`

## SST Encoding

| Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96592 | 614,404 | 1.172 | 27.31x | 17,391,620 | 0.96x | 2.5124% | 2.6270% | 0.0000% | 0.3737% |

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\00_depth.png` | 0.032 |
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\01_compact-pcf.png` | 0.020 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\diff_depth_vs_compact-pcf.png` | 0.0075% | 0.2379% | 19.2157% | 0.2380% | 52.47 dB | 0.2195 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\diff_depth_vs_compact-pcf.png` | 0.0075% | 0.2379% | 19.2157% | 0.2380% | 52.47 dB |
