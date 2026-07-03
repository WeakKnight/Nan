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
| depth | `static_shadow_compare_bistro_64_modes\00_depth.png` | 0.030 |
| compact | `static_shadow_compare_bistro_64_modes\01_compact.png` | 0.005 |
| compact-pcf | `static_shadow_compare_bistro_64_modes\02_compact-pcf.png` | 0.011 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact | `static_shadow_compare_bistro_64_modes\diff_depth_vs_compact.png` | 0.2223% | 2.1075% | 43.5294% | 2.2705% | 33.52 dB | 0.2175 |
| compact-pcf | `static_shadow_compare_bistro_64_modes\diff_depth_vs_compact-pcf.png` | 0.0552% | 0.5310% | 13.3333% | 1.7090% | 45.50 dB | 0.2180 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | compact | `static_shadow_compare_bistro_64_modes\diff_depth_vs_compact.png` | 0.2223% | 2.1075% | 43.5294% | 2.2705% | 33.52 dB |
| depth | compact-pcf | `static_shadow_compare_bistro_64_modes\diff_depth_vs_compact-pcf.png` | 0.0552% | 0.5310% | 13.3333% | 1.7090% | 45.50 dB |
| compact | compact-pcf | `static_shadow_compare_bistro_64_modes\diff_compact_vs_compact-pcf.png` | 0.1947% | 1.8171% | 44.3137% | 2.4170% | 34.81 dB |
