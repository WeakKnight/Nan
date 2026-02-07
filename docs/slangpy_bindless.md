# SlangPy Bindless 资源使用指南

## 概述

Bindless 是一种现代 GPU 编程技术，允许着色器通过描述符句柄（Descriptor Handle）直接访问资源，而无需在着色器参数中显式绑定每个资源。这使得可以在运行时动态访问大量资源。

**支持情况**：
- **Bindless Buffers**: D3D12, Vulkan
- **Bindless Textures**: D3D12, Vulkan, CUDA

## 检测 Bindless 支持

```python
import slangpy as spy

device = spy.Device()
if device.has_feature(spy.Feature.bindless):
    print("Bindless is supported!")
```

## Bindless Texture 示例

### Slang 着色器

```slang
struct TextureInfo
{
    Texture2D<float>.Handle texture;    // Bindless 纹理句柄
    SamplerState.Handle sampler;        // Bindless 采样器句柄
    float2 uv;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(
    uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<TextureInfo> texture_infos,
    RWStructuredBuffer<float> results)
{
    uint index = tid.x;
    TextureInfo info = texture_infos[index];
    // 通过句柄采样纹理
    results[index] = info.texture.SampleLevel(info.sampler, info.uv, 0);
}
```

### Python 代码

```python
import slangpy as spy
import numpy as np

device = spy.Device(type=spy.DeviceType.d3d12)

# 1. 加载模块并创建 kernel
module = device.load_module("test_bindless_texture.slang")
program = device.link_program(
    modules=[module],
    entry_points=[module.entry_point("compute_main")]
)
kernel = device.create_compute_kernel(program)

TEXTURE_COUNT = 8

# 2. 创建采样器
sampler_linear = device.create_sampler()
sampler_point = device.create_sampler(
    min_filter=spy.TextureFilteringMode.point,
    mag_filter=spy.TextureFilteringMode.point,
)

# 3. 创建纹理并获取视图
textures = []
texture_views = []
for i in range(TEXTURE_COUNT):
    texture = device.create_texture(
        width=2, height=1,
        format=spy.Format.r32_float,
        usage=spy.TextureUsage.shader_resource,
        data=np.array([i, i + 1], dtype=np.float32),
    )
    textures.append(texture)
    texture_views.append(texture.create_view())

# 4. 获取 TextureInfo 结构体布局
texture_info_layout = module.layout.get_type_layout(
    module.layout.find_type_by_name("StructuredBuffer<TextureInfo>")
).element_type_layout

# 5. 创建存储 TextureInfo 的 buffer
texture_infos_buffer = device.create_buffer(
    size=TEXTURE_COUNT * texture_info_layout.stride,
    usage=spy.BufferUsage.shader_resource,
)

results_buffer = device.create_buffer(
    size=TEXTURE_COUNT * 4,
    usage=spy.BufferUsage.unordered_access,
)

# 6. 使用 BufferCursor 填充 bindless 描述符句柄
c = spy.BufferCursor(texture_info_layout, texture_infos_buffer, load_before_write=False)
for i in range(TEXTURE_COUNT):
    c[i].texture = texture_views[i].descriptor_handle_ro  # 纹理只读句柄
    c[i].sampler = sampler_linear.descriptor_handle       # 采样器句柄
    c[i].uv = spy.float2(0.5)
c.apply()

# 7. 调度 kernel
kernel.dispatch(
    thread_count=[TEXTURE_COUNT, 1, 1],
    texture_infos=texture_infos_buffer,
    results=results_buffer,
)

# 8. 读取结果
results = results_buffer.to_numpy().view(np.float32)
```

## Bindless Buffer 示例

### Slang 着色器

```slang
struct BufferInfo
{
    StructuredBuffer<float>.Handle ro_buffer;    // 只读 buffer 句柄
    RWStructuredBuffer<float>.Handle rw_buffer;  // 读写 buffer 句柄
    uint offset;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(
    uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<BufferInfo> buffer_infos,
    RWStructuredBuffer<float> results)
{
    uint index = tid.x;
    BufferInfo info = buffer_infos[index];

    // 通过句柄读取只读 buffer
    float value = info.ro_buffer[info.offset];

    // 通过句柄写入读写 buffer
    info.rw_buffer[info.offset] = value + 100.0;

    results[index] = value;
}
```

### Python 代码

```python
import slangpy as spy
import numpy as np

device = spy.Device(type=spy.DeviceType.d3d12)

module = device.load_module("test_bindless_buffer.slang")
program = device.link_program(
    modules=[module],
    entry_points=[module.entry_point("compute_main")]
)
kernel = device.create_compute_kernel(program)

BUFFER_COUNT = 6

# 创建只读 buffer
ro_buffers = []
for i in range(BUFFER_COUNT):
    buffer = device.create_buffer(
        size=4 * 4,  # 4 floats
        usage=spy.BufferUsage.shader_resource,
        data=np.array([i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3], dtype=np.float32),
    )
    ro_buffers.append(buffer)

# 创建读写 buffer
rw_buffers = []
for i in range(BUFFER_COUNT):
    buffer = device.create_buffer(
        size=4 * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.zeros(4, dtype=np.float32),
    )
    rw_buffers.append(buffer)

# 获取 BufferInfo 布局
buffer_info_layout = module.layout.get_type_layout(
    module.layout.find_type_by_name("StructuredBuffer<BufferInfo>")
).element_type_layout

buffer_infos_buffer = device.create_buffer(
    size=BUFFER_COUNT * buffer_info_layout.stride,
    usage=spy.BufferUsage.shader_resource,
)

results_buffer = device.create_buffer(
    size=BUFFER_COUNT * 4,
    usage=spy.BufferUsage.unordered_access,
)

# 填充 bindless buffer 句柄
c = spy.BufferCursor(buffer_info_layout, buffer_infos_buffer, load_before_write=False)
for i in range(BUFFER_COUNT):
    c[i].ro_buffer = ro_buffers[i].descriptor_handle_ro  # 只读句柄
    c[i].rw_buffer = rw_buffers[i].descriptor_handle_rw  # 读写句柄
    c[i].offset = i % 4
c.apply()

kernel.dispatch(
    thread_count=[BUFFER_COUNT, 1, 1],
    buffer_infos=buffer_infos_buffer,
    results=results_buffer,
)
```

## 关键 API 说明

### 描述符句柄属性

| 资源类型 | 属性 | 说明 |
|---------|------|------|
| `Buffer` | `descriptor_handle_ro` | 获取只读访问的 bindless 句柄 |
| `Buffer` | `descriptor_handle_rw` | 获取读写访问的 bindless 句柄 |
| `TextureView` | `descriptor_handle_ro` | 获取只读访问的 bindless 句柄 |
| `TextureView` | `descriptor_handle_rw` | 获取读写访问的 bindless 句柄 |
| `Sampler` | `descriptor_handle` | 获取采样器的 bindless 句柄 |

### Slang 句柄类型

| Slang 类型 | 用途 |
|-----------|------|
| `Texture2D<T>.Handle` | 2D 纹理句柄 |
| `SamplerState.Handle` | 采样器句柄 |
| `StructuredBuffer<T>.Handle` | 只读结构化 buffer 句柄 |
| `RWStructuredBuffer<T>.Handle` | 读写结构化 buffer 句柄 |

### BufferCursor

`BufferCursor` 用于方便地填充结构化数据到 buffer 中：

```python
cursor = spy.BufferCursor(type_layout, buffer, load_before_write=False)
cursor[index].field_name = value
cursor.apply()  # 将数据写入 buffer
```

## 注意事项

1. **检测支持**：使用前必须检查 `device.has_feature(spy.Feature.bindless)`
2. **平台限制**：
   - Bindless Buffers 支持 D3D12 和 Vulkan
   - Bindless Textures 支持 D3D12、Vulkan 和 CUDA
3. **纹理视图**：纹理需要通过 `texture.create_view()` 创建视图后才能获取句柄
4. **Buffer 用途**：
   - 只读 buffer 需要 `BufferUsage.shader_resource`
   - 读写 buffer 需要 `BufferUsage.shader_resource | BufferUsage.unordered_access`
