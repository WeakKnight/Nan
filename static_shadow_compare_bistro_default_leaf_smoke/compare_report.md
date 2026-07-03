# Static Shadow Runtime Compare

- Scene: `E:\GitHub\niagara_bistro\bistro.gltf`
- Resolution: `64x64`
- Frames: `1`
- Shadow resolution: `256`
- Reference mode: `depth`
- Bake time: `0.004s`
- Encode time: `1.717s`
- SST encoded: `True`
- SST profile: `Dual Visible`
- SST tile/leaf: `128/2`
- SST packed ratio: `5.22x`
- SST packed bpt: `6.129`
- SST packed+decomp ratio: `0.84x`

## SST Encoding

| Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Shadow > 1 bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6692 | 50,212 | 6.129 | 5.22x | 312,356 | 0.84x | 1.6864% | 3.6255% | 0.0000% | 2.4292% |

## Images

| Mode | Output | Render seconds |
|---|---|---:|
| depth | `static_shadow_compare_bistro_default_leaf_smoke\00_depth.png` | 0.028 |
| compact-pcf | `static_shadow_compare_bistro_default_leaf_smoke\01_compact-pcf.png` | 0.014 |

## Diffs

| Candidate | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR | Candidate mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact-pcf | `static_shadow_compare_bistro_default_leaf_smoke\diff_depth_vs_compact-pcf.png` | 0.0369% | 0.3974% | 16.0784% | 1.3184% | 48.02 dB | 0.2183 |

## Pairwise Diffs

| A | B | Diff preview | Mean abs | RMSE | Max abs | Changed px | PSNR |
|---|---|---|---:|---:|---:|---:|---:|
| depth | compact-pcf | `static_shadow_compare_bistro_default_leaf_smoke\diff_depth_vs_compact-pcf.png` | 0.0369% | 0.3974% | 16.0784% | 1.3184% | 48.02 dB |
