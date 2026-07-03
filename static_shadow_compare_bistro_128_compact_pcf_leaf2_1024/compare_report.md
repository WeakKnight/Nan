# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `128x128`
- Frames: `1`
- Shadow resolution: `1024`
- Reference mode: `depth`
- Bake time: `0.004s`
- Encode time: `18.224s`
- SST encoded: `True`
- SST profile: `Dual Visible`
- SST tile/leaf: `128/2`
- SST packed ratio: `15.05x`
- SST packed bpt: `2.127`
- SST packed+decomp ratio: `0.94x`

## SST Encoding

| Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41800 | 278,756 | 2.127 | 15.05x | 4,473,060 | 0.94x | 2.3723% | 2.5915% | 0.0000% | 0.6459% |

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_128_compact_pcf_leaf2_1024\00_depth.png` | 0.031 |
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2_1024\01_compact-pcf.png` | 0.013 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2_1024\diff_depth_vs_compact-pcf.png` | 0.0151% | 0.2531% | 13.3333% | 0.6348% | 51.93 dB | 0.2194 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | compact-pcf | `static_shadow_compare_bistro_128_compact_pcf_leaf2_1024\diff_depth_vs_compact-pcf.png` | 0.0151% | 0.2531% | 13.3333% | 0.6348% | 51.93 dB |
