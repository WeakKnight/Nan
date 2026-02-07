# TextureLoader

`TextureLoader` 是一个用于从 Bitmap 或图像文件加载纹理的工具类。

## 创建 TextureLoader

```python
import slangpy as spy

device = spy.Device()
loader = spy.TextureLoader(device)
```

## 加载选项 (Options)

`TextureLoader.Options` 提供以下配置选项：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `load_as_normalized` | bool | `True` | 将 8/16 位整数数据加载为归一化资源格式 |
| `load_as_srgb` | bool | `True` | 如果 Bitmap 是带 sRGB gamma 的 8 位 RGBA，使用 `Format.rgba8_unorm_srgb` 格式 |
| `extend_alpha` | bool | `True` | 如果 RGB 纹理格式不可用，扩展 RGB 为 RGBA |
| `allocate_mips` | bool | `False` | 为纹理分配 mip levels |
| `generate_mips` | bool | `False` | 生成 mip levels（会自动添加 `TextureUsage.render_target`） |
| `usage` | TextureUsage | `shader_resource` | 纹理的资源用途标志 |

### 使用选项

```python
# 方式 1：使用 Options 对象
options = spy.TextureLoader.Options()
options.generate_mips = True
options.load_as_srgb = True

# 方式 2：使用字典
options = {
    "generate_mips": True,
    "load_as_srgb": True,
}
```

## API 方法

### load_texture

从单个 Bitmap 或图像文件加载纹理。

```python
# 从文件加载
texture = loader.load_texture("path/to/image.png")

# 从文件加载并指定选项
texture = loader.load_texture("path/to/image.png", options={"generate_mips": True})

# 从 Bitmap 加载
bitmap = spy.Bitmap("path/to/image.png")
texture = loader.load_texture(bitmap)
```

### load_textures

从多个 Bitmap 或图像文件批量加载纹理。

```python
# 从多个文件加载
paths = ["image1.png", "image2.png", "image3.png"]
textures = loader.load_textures(paths)

# 从多个 Bitmap 加载
bitmaps = [spy.Bitmap(p) for p in paths]
textures = loader.load_textures(bitmaps)
```

### load_texture_array

从多个 Bitmap 或图像文件加载为纹理数组（Texture Array）。

> **注意**：所有图像必须具有相同的格式和尺寸。

```python
# 从多个文件加载为纹理数组
paths = ["frame0.png", "frame1.png", "frame2.png", "frame3.png"]
texture_array = loader.load_texture_array(paths)

# 从多个 Bitmap 加载为纹理数组
bitmaps = [spy.Bitmap(p) for p in paths]
texture_array = loader.load_texture_array(bitmaps)
```

## 完整示例

```python
import slangpy as spy

# 创建设备和加载器
device = spy.Device()
loader = spy.TextureLoader(device)

# 加载单个纹理并生成 mipmaps
texture = loader.load_texture(
    "diffuse.png",
    options={
        "generate_mips": True,
        "load_as_srgb": True,
        "usage": spy.TextureUsage.shader_resource,
    }
)

# 批量加载多个纹理
texture_paths = ["tex0.png", "tex1.png", "tex2.png"]
textures = loader.load_textures(texture_paths)

# 加载为纹理数组（适用于动画帧、立方体贴图面等）
animation_frames = ["frame0.png", "frame1.png", "frame2.png", "frame3.png"]
animation_texture = loader.load_texture_array(
    animation_frames,
    options={"generate_mips": True}
)

print(f"Loaded texture: {texture.width}x{texture.height}, mip_count={texture.mip_count}")
print(f"Loaded {len(textures)} textures")
print(f"Texture array: {animation_texture.width}x{animation_texture.height}, array_length={animation_texture.array_length}")
```

## 支持的图像格式

TextureLoader 通过 `Bitmap` 类加载图像，支持常见的图像格式，包括：

- PNG
- JPEG
- BMP
- EXR（HDR）
- 其他 stb_image 支持的格式
