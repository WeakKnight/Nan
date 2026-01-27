import numpy as np
import slangpy as spy

def fill_neighbor_offset_array(neighbor_offset_count: int) -> np.ndarray:
    R = 250
    phi2 = 1.0 / 1.3247179572447
    u = 0.5
    v = 0.5

    out = np.empty(neighbor_offset_count * 2, dtype = np.float32)
    num = 0

    while num < out.size:
        u += phi2
        v += phi2 * phi2
        if u >= 1.0:
            u -= 1.0
        if v >= 1.0:
            v -= 1.0

        r_sq = (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)
        if r_sq > 0.25:
            continue

        out[num] = u - 0.5
        out[num + 1] = v - 0.5

        num += 2

    return out

class LowDiscrepancyDiskPattern:
    def __init__(self, device: spy.Device, sample_count: int = 8192):
        buffer_data = fill_neighbor_offset_array(sample_count)
        self.neighbor_offset_buffer = device.create_buffer(usage = spy.BufferUsage.shader_resource, label = "neighbor_offset_buffer", struct_size = 8, data = buffer_data)
        self.neighbor_offset_mask = sample_count - 1
