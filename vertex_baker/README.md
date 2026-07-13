# Vertex Baker

## Native SlangPy Viewer

The interactive viewer uses Slang compute passes with inline HWRT `RayQuery` to generate a stable GBuffer. Its default PBR view loads core glTF metallic-roughness materials and uses a GPU-generated Filament-style IBL from `bloem_field_sunrise_2k.hdr`.

Inspect an existing visibility bake without rebaking:

```powershell
python vertex_baker/main.py `
  --mode interactive `
  --asset vertex_baker/glTF/Lantern.gltf `
  --viewer-data vertex_baker/out/lantern_visibility_vertex_cones_256.visibility.npz `
  --envmap vertex_baker/bloem_field_sunrise_2k.hdr
```

Controls:

- Left drag: orbit
- Right or middle drag: pan
- Mouse wheel: zoom
- Left click: select the nearest visible vertex on the clicked mesh within `Pick radius`
- `Pick radius`: screen-space selection radius in pixels; clicks with no nearby vertex clear the selection
- `F2`: save the viewer output to `--output`
- `Esc`: close

The PBR controls expose display exposure, environment rotation/background, and optional PMR visibility-cone occlusion. The cone contributes separate per-pixel diffuse AO and view-dependent specular occlusion; material AO remains diffuse-only. Use `--no-apply-visibility` to disable both cone terms in automated captures. The `PMR diffuse AO` and `PMR specular occlusion` views inspect the evaluated terms directly.

Use `--viewer-backend html` to preserve the legacy HTML export. `--viewer-max-frames N` and `--viewer-capture-on-exit` are useful for automated GPU smoke tests.

## GBuffer

| Resource | Format | Channels |
|---|---|---|
| Normal/roughness | `rgba16_float` | World normal, roughness |
| Albedo/metallic | `rgba16_float` | Base color, metallic |
| Emissive/occlusion | `rgba16_float` | Emissive radiance, material AO |
| Baked value | `rgba16_float` | Interpolated vertex bake |
| Visibility cone | `rgba16_float` | Interpolated world-space direction, normalized aperture |
| Cone parameters | `rg16_float` | Aperture, scale |
| Linear depth | `r32_float` | Positive view-space depth |
| IDs | `rgba32_uint` | Triangle, mesh, nearest vertex, valid flag |

World position can be reconstructed from linear depth and the camera matrices. This layout is intended to feed later image-based-lighting and material-debug passes without changing the primary-ray pass.

At startup the viewer converts the equirectangular HDR to a cubemap, projects diffuse irradiance to SH9, generates five GGX-prefiltered specular mips, integrates a 128px RG16F DFG LUT, and builds PMR's 16x16x16 cone-BRDF specular-occlusion atlas with 256 GGX VNDF samples per texel. All preprocessing runs on the active SlangPy device and is retained for the viewer lifetime.
