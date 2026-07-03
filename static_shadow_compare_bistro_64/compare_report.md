# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `64x64`
- Frames: `1`
- Shadow resolution: `64`
- Reference mode: `depth`

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_64\00_depth.png` | 16.542 |
| compact-pcf | `static_shadow_compare_bistro_64\01_compact-pcf.png` | 14.900 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact-pcf | `static_shadow_compare_bistro_64\diff_depth_vs_compact-pcf.png` | 0.0552% | 0.5310% | 13.3333% | 1.7090% | 45.50 dB | 0.2180 |
