# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `64x64`
- Frames: `1`
- Shadow resolution: `64`
- Reference mode: `depth`
- Bake time: `0.151s`
- Encode time: `0.000s`
- SST encoded: `True`

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_64_decompressed_relaxed\00_depth.png` | 0.029 |
| decompressed | `static_shadow_compare_bistro_64_decompressed_relaxed\01_decompressed.png` | 0.012 |
| compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed\02_compact-pcf.png` | 0.008 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| decompressed | `static_shadow_compare_bistro_64_decompressed_relaxed\diff_depth_vs_decompressed.png` | 0.0032% | 0.0863% | 3.5294% | 0.1465% | 61.28 dB | 0.2183 |
| compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed\diff_depth_vs_compact-pcf.png` | 0.0545% | 0.5282% | 13.3333% | 1.6846% | 45.54 dB | 0.2180 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | decompressed | `static_shadow_compare_bistro_64_decompressed_relaxed\diff_depth_vs_decompressed.png` | 0.0032% | 0.0863% | 3.5294% | 0.1465% | 61.28 dB |
| depth | compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed\diff_depth_vs_compact-pcf.png` | 0.0545% | 0.5282% | 13.3333% | 1.6846% | 45.54 dB |
| decompressed | compact-pcf | `static_shadow_compare_bistro_64_decompressed_relaxed\diff_decompressed_vs_compact-pcf.png` | 0.0546% | 0.5270% | 13.3333% | 1.6846% | 45.56 dB |
