# Static Shadow Runtime Compare Summary

- Scenes: `E:\GitHub\niagara_bistro\bistro.gltf`
- Sources: `static_shadow_compare_bistro_128_compact_pcf_leaf2\compare_report.json`, `static_shadow_compare_bistro_128_compact_pcf_leaf2_1024\compare_report.json`, `static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\compare_report.json`, `static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\compare_report.json`

## Runtime Recommendations

| Constraint | Source | Shadow res | Profile | Tile | Leaf | Mode | Packed | Packed bpt | PCF3 MAE | Mean abs | RMSE | Changed px | PSNR |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| best_quality | static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\compare_report.json | 2048 | Dual Visible | 128 | 2 | compact-pcf | 27.31x | 1.172 | 2.5124% | 0.0075% | 0.2379% | 0.2380% | 52.47 dB |
| max_compression_mean_abs_le_0_02_percent | static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\compare_report.json | 2048 | Dual Visible | 128 | 2 | compact-pcf | 27.31x | 1.172 | 2.5124% | 0.0075% | 0.2379% | 0.2380% | 52.47 dB |
| max_compression_changed_px_le_0_5_percent | static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\compare_report.json | 2048 | Dual Visible | 128 | 2 | compact-pcf | 27.31x | 1.172 | 2.5124% | 0.0075% | 0.2379% | 0.2380% | 52.47 dB |
| max_compression_mean_abs_le_0_05_percent | static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\compare_report.json | 2048 | Dual Relaxed Visible | 128 | 2 | compact-pcf | 34.92x | 0.916 | 5.1311% | 0.0372% | 0.7482% | 0.5005% | 42.52 dB |
| max_compression_changed_px_le_1_percent | static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\compare_report.json | 2048 | Dual Relaxed Visible | 128 | 2 | compact-pcf | 34.92x | 0.916 | 5.1311% | 0.0372% | 0.7482% | 0.5005% | 42.52 dB |

## Compact Runtime Quality

| Source | Image res | Shadow res | Profile | Tile | Leaf | Candidate | Packed | Packed bpt | Packed+decomp ratio | PCF3 MAE | Vis mismatch | False lit | Mean abs | RMSE | Changed px | PSNR | Candidate render | Encode time |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_shadow_compare_bistro_128_compact_pcf_leaf2\compare_report.json | 128x128 | 512 | Dual Visible | 128 | 2 | compact-pcf | 8.68x | 3.688 | 0.90x | 2.1255% | 3.1948% | 0.0000% | 0.0209% | 0.2569% | 1.0559% | 51.80 dB | 0.015s | 5.429s |
| static_shadow_compare_bistro_128_compact_pcf_leaf2_1024\compare_report.json | 128x128 | 1024 | Dual Visible | 128 | 2 | compact-pcf | 15.05x | 2.127 | 0.94x | 2.3723% | 2.5915% | 0.0000% | 0.0151% | 0.2531% | 0.6348% | 51.93 dB | 0.013s | 18.224s |
| static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\compare_report.json | 128x128 | 2048 | Dual Visible | 128 | 2 | compact-pcf | 27.31x | 1.172 | 0.96x | 2.5124% | 2.6270% | 0.0000% | 0.0075% | 0.2379% | 0.2380% | 52.47 dB | 0.020s | 60.990s |
| static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\compare_report.json | 128x128 | 2048 | Dual Relaxed Visible | 128 | 2 | compact-pcf | 34.92x | 0.916 | 0.97x | 5.1311% | 6.1282% | 0.0000% | 0.0372% | 0.7482% | 0.5005% | 42.52 dB | 0.014s | 61.243s |

## Storage

| Source | Shadow res | Profile | Tile | Leaf | Nodes | Packed bytes | Packed bpt | Packed ratio | Packed+decomp bytes | Packed+decomp ratio | Fixed64 bytes | Fixed64 ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_shadow_compare_bistro_128_compact_pcf_leaf2\compare_report.json | 512 | Dual Visible | 128 | 2 | 17452 | 120,848 | 3.688 | 8.68x | 1,169,424 | 0.90x | 139,680 | 7.51x |
| static_shadow_compare_bistro_128_compact_pcf_leaf2_1024\compare_report.json | 1024 | Dual Visible | 128 | 2 | 41800 | 278,756 | 2.127 | 15.05x | 4,473,060 | 0.94x | 334,656 | 12.53x |
| static_shadow_compare_bistro_128_compact_pcf_leaf2_2048\compare_report.json | 2048 | Dual Visible | 128 | 2 | 96592 | 614,404 | 1.172 | 27.31x | 17,391,620 | 0.96x | 773,760 | 21.68x |
| static_shadow_compare_bistro_128_compact_pcf_relaxed_leaf2_2048\compare_report.json | 2048 | Dual Relaxed Visible | 128 | 2 | 77516 | 480,404 | 0.916 | 34.92x | 17,257,620 | 0.97x | 621,152 | 27.01x |
