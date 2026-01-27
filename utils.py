import OpenEXR
import Imath
import numpy as np
import slangpy as spy

def load_exr(file_path):
    # Open EXR file
    exr_file = OpenEXR.InputFile(file_path)

    # Get data window to determine image dimensions
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    print("header", header)

    # Define pixel data type, using FLOAT (32-bit float)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read channel data
    # Assuming EXR file contains "R", "G", "B" channels, add "A" if Alpha is needed
    channels = ["R", "G", "B"]
    data = {}
    for c in channels:
        # Read channel data, returns byte stream
        ch_str = exr_file.channel(c, pt)
        # Convert byte data to numpy array with dtype float32
        data[c] = np.frombuffer(ch_str, dtype=np.float32)
        # Reshape to (height, width)
        data[c] = np.reshape(data[c], (width, height))

    # Stack channel arrays along last axis, resulting in shape (height, width, 3)
    img = np.stack([data[c] for c in channels], axis=-1)
    img = img.astype(np.float32)
    return img

def load_exr_as_slang_texture(device, file_path):
    probe_data = load_exr(file_path)
    img_w = probe_data.shape[0]
    img_h = probe_data.shape[1]
    alpha_channel = np.ones((img_w, img_h, 1), dtype=probe_data.dtype)
    probe_data = np.concatenate([probe_data, alpha_channel], axis=2)
    probe_data = probe_data.reshape(-1, 1, 1)
    radiance_map = device.create_texture(width= img_w, height = img_h, format = spy.Format.rgba32_float,
                                  usage= spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                                  data = probe_data)
    return radiance_map;

def save_buffer(device: spy.Device, buffer: spy.Buffer, file_path):
    buffer.to_numpy()