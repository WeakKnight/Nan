# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `128x128`
- Frames: `1`
- Shadow resolution: `512`
- Reference mode: `depth`
- Bake time: `0.005s`
- Encode time: `5.429s`
- SST encoded: `True`
- SST profile: `Dual Visible`
- SST tile/leaf: `128/2`
- SST packed ratio: `8.68x`
- SST packed bpt: `3.688`
- SST packed+decomp ratio: `0.90x`

## SST Encoding

| Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17452 | 120,848 | 3.688 | 8.68x | 1,169,424 | 0.90x | 2.1255% | 3.1948% | 0.0000% | 1.1528% |

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_128_compact_pcf_leaf2\00_depth.png` | 0.032 |
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2\01_compact-pcf.png` | 0.015 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2\diff_depth_vs_compact-pcf.png` | 0.0209% | 0.2569% | 10.9804% | 1.0559% | 51.80 dB | 0.2191 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2\diff_depth_vs_compact-pcf.png` | 0.0209% | 0.2569% | 10.9804% | 1.0559% | 51.80 dB |
