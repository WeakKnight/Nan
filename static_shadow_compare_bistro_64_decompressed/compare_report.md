# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `64x64`
- Frames: `1`
- Shadow resolution: `64`
- Reference mode: `depth`
- Bake time: `0.152s`
- Encode time: `0.000s`
- SST encoded: `True`

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_64_decompressed\00_depth.png` | 0.027 |
| decompressed | `static_shadow_compare_bistro_64_decompressed\01_decompressed.png` | 0.013 |
| compact-pcf | `static_shadow_compare_bistro_64_decompressed\02_compact-pcf.png` | 0.005 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| decompressed | `static_shadow_compare_bistro_64_decompressed\diff_depth_vs_decompressed.png` | 0.0025% | 0.0807% | 3.5294% | 0.0977% | 61.87 dB | 0.2183 |
| compact-pcf | `static_shadow_compare_bistro_64_decompressed\diff_depth_vs_compact-pcf.png` | 0.0552% | 0.5310% | 13.3333% | 1.7090% | 45.50 dB | 0.2180 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | decompressed | `static_shadow_compare_bistro_64_decompressed\diff_depth_vs_decompressed.png` | 0.0025% | 0.0807% | 3.5294% | 0.0977% | 61.87 dB |
| depth | compact-pcf | `static_shadow_compare_bistro_64_decompressed\diff_depth_vs_compact-pcf.png` | 0.0552% | 0.5310% | 13.3333% | 1.7090% | 45.50 dB |
| decompressed | compact-pcf | `static_shadow_compare_bistro_64_decompressed\diff_decompressed_vs_compact-pcf.png` | 0.0553% | 0.5273% | 13.3333% | 1.7334% | 45.56 dB |
