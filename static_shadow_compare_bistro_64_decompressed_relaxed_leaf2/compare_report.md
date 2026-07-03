# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `64x64`
- Frames: `1`
- Shadow resolution: `64`
- Reference mode: `depth`
- Bake time: `0.002s`
- Encode time: `0.137s`
- SST encoded: `True`

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\00_depth.png` | 0.028 |
| decompressed | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\01_decompressed.png` | 0.018 |
| compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\02_compact-pcf.png` | 0.004 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| decompressed | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\diff_depth_vs_decompressed.png` | 0.0218% | 0.2564% | 6.2745% | 0.9277% | 51.82 dB | 0.2181 |
| compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\diff_depth_vs_compact-pcf.png` | 0.0541% | 0.5116% | 12.9412% | 1.7578% | 45.82 dB | 0.2179 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | decompressed | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\diff_depth_vs_decompressed.png` | 0.0218% | 0.2564% | 6.2745% | 0.9277% | 51.82 dB |
| depth | compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\diff_depth_vs_compact-pcf.png` | 0.0541% | 0.5116% | 12.9412% | 1.7578% | 45.82 dB |
| decompressed | compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed_leaf2\diff_decompressed_vs_compact-pcf.png` | 0.0507% | 0.4926% | 10.1961% | 1.6602% | 46.15 dB |
