import slangpy as spy
from typing import Optional

class AtmosphereTransmittanceLUT:
    """
    Generates the atmospheric transmittance lookup table (LUT).
    
    The LUT stores transmittance values from different heights and sun zenith angles
    through the atmosphere. This is used for realistic atmospheric scattering.
    
    LUT dimensions: 256 x 64 (width x height)
    Format: RGBA32F (float4)
    """
    
    # LUT resolution from atmosphere.slang
    LUT_WIDTH = 256
    LUT_HEIGHT = 64
    
    def __init__(self, device: spy.Device):
        self.device = device
        
        # Load the compute shader program
        self.program = self.device.load_program(
            "atmosphere.slang", 
            ["transmittance_lut_compute"]
        )
        self.kernel = self.device.create_compute_kernel(self.program)
        
        # Create the transmittance LUT texture
        self.transmittance_lut: Optional[spy.Texture] = None
        self._create_lut_texture()
    
    def _create_lut_texture(self):
        """Creates or recreates the transmittance LUT texture."""
        self.transmittance_lut = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=self.LUT_WIDTH,
            height=self.LUT_HEIGHT,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="atmosphere_transmittance_lut",
        )
    
    def generate(self, command_encoder: spy.CommandEncoder):
        """
        Dispatches the compute shader to generate the transmittance LUT.
        
        This computes the atmospheric transmittance for each (height, sun_zenith_angle)
        pair and stores it in the LUT texture.
        
        Args:
            command_encoder: Command encoder to record the compute dispatch
        """
        self.kernel.dispatch(
            thread_count=[self.LUT_WIDTH, self.LUT_HEIGHT, 1],
            vars={
                "g_transmittance_lut": self.transmittance_lut,
            },
            command_encoder=command_encoder,
        )
    
    def get_texture(self) -> spy.Texture:
        """Returns the generated transmittance LUT texture for use in other shaders."""
        return self.transmittance_lut


class AtmosphereMultiScatteringLUT:
    """
    Generates the atmospheric multiple scattering lookup table (LUT).
    
    The LUT stores pre-computed multiple scattering values (Psi_ms from the paper)
    for different heights and sun zenith angles. This approximates higher-order
    scattering contributions.
    
    LUT dimensions: 32 x 32 (width x height)
    Format: RGBA32F (float4)
    
    Requires: Transmittance LUT must be generated first
    """
    
    # LUT resolution from atmosphere.slang
    LUT_WIDTH = 32
    LUT_HEIGHT = 32
    
    def __init__(self, device: spy.Device):
        self.device = device
        
        # Load the compute shader program
        self.program = self.device.load_program(
            "atmosphere.slang", 
            ["multiscatt_lut_compute"]
        )
        self.kernel = self.device.create_compute_kernel(self.program)
        
        # Create sampler state for texture sampling
        self.sampler = self.device.create_sampler(
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
        )
        
        # Create the multiple scattering LUT texture
        self.multiscatt_lut: Optional[spy.Texture] = None
        self._create_lut_texture()
    
    def _create_lut_texture(self):
        """Creates or recreates the multiple scattering LUT texture."""
        self.multiscatt_lut = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=self.LUT_WIDTH,
            height=self.LUT_HEIGHT,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="atmosphere_multiscatt_lut",
        )
    
    def generate(self, command_encoder: spy.CommandEncoder, transmittance_lut: spy.Texture):
        """
        Dispatches the compute shader to generate the multiple scattering LUT.
        
        This computes the multiple scattering approximation for each (height, sun_zenith_angle)
        pair by integrating over many ray directions and path segments.
        
        Args:
            command_encoder: Command encoder to record the compute dispatch
            transmittance_lut: The transmittance LUT texture (must be pre-generated)
        """
        self.kernel.dispatch(
            thread_count=[self.LUT_WIDTH, self.LUT_HEIGHT, 1],
            vars={
                "g_transmittance_lut_input": transmittance_lut,
                "g_sampler": self.sampler,
                "g_multiscatt_lut": self.multiscatt_lut,
            },
            command_encoder=command_encoder,
        )
    
    def get_texture(self) -> spy.Texture:
        """Returns the generated multiple scattering LUT texture for use in other shaders."""
        return self.multiscatt_lut


class AtmosphereSkyViewLUT:
    """
    Generates the atmospheric sky-view lookup table (LUT).
    
    The LUT stores the pre-computed sky appearance in a latitude/azimuth format
    with non-linear mapping to get more resolution near the horizon. This is used
    for efficient sky rendering.
    
    LUT dimensions: 200 x 200 (width x height)
    Format: RGBA32F (float4)
    
    Requires: Transmittance LUT and Multiple Scattering LUT must be generated first
    """
    
    # LUT resolution from atmosphere.slang
    LUT_WIDTH = 200
    LUT_HEIGHT = 200
    
    def __init__(self, device: spy.Device):
        self.device = device
        
        # Load the compute shader program
        self.program = self.device.load_program(
            "atmosphere.slang", 
            ["sky_view_lut_compute"]
        )
        self.kernel = self.device.create_compute_kernel(self.program)
        
        # Create sampler state for texture sampling
        self.sampler = self.device.create_sampler(
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
        )
        
        # Create the sky view LUT texture
        self.sky_view_lut: Optional[spy.Texture] = None
        self._create_lut_texture()
    
    def _create_lut_texture(self):
        """Creates or recreates the sky view LUT texture."""
        self.sky_view_lut = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=self.LUT_WIDTH,
            height=self.LUT_HEIGHT,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="atmosphere_sky_view_lut",
        )
    
    def generate(
        self, 
        command_encoder: spy.CommandEncoder, 
        transmittance_lut: spy.Texture,
        multiscatt_lut: spy.Texture,
        sun_direction: tuple = (0.0, 0.5, -0.866)
    ):
        """
        Dispatches the compute shader to generate the sky view LUT.
        
        This computes the sky appearance by raymarching through the atmosphere
        for each (azimuth, altitude) direction, incorporating both single and
        multiple scattering.
        
        Args:
            command_encoder: Command encoder to record the compute dispatch
            transmittance_lut: The transmittance LUT texture (must be pre-generated)
            multiscatt_lut: The multiple scattering LUT texture (must be pre-generated)
            sun_direction: Sun direction vector in Y-up coordinate system (x, y, z)
        """
        self.kernel.dispatch(
            thread_count=[self.LUT_WIDTH, self.LUT_HEIGHT, 1],
            vars={
                "g_transmittance_lut_input_sky": transmittance_lut,
                "g_multiscatt_lut_input": multiscatt_lut,
                "g_sampler_sky": self.sampler,
                "g_sky_view_lut": self.sky_view_lut,
                "g_sun_direction": sun_direction,
            },
            command_encoder=command_encoder,
        )
    
    def get_texture(self) -> spy.Texture:
        """Returns the generated sky view LUT texture for use in other shaders."""
        return self.sky_view_lut


# Example usage:
if __name__ == "__main__":
    import pathlib
    
    # Create device with include paths
    device = spy.create_device(
        include_paths=[
            pathlib.Path(__file__).parent.absolute(),
        ]
    )
    
    # Create all three LUT generators
    transmittance_gen = AtmosphereTransmittanceLUT(device)
    multiscatt_gen = AtmosphereMultiScatteringLUT(device)
    sky_view_gen = AtmosphereSkyViewLUT(device)
    
    # Create command encoder
    command_encoder = device.create_command_encoder()
    
    # Generate the transmittance LUT first (required by others)
    print("Generating transmittance LUT...")
    transmittance_gen.generate(command_encoder)
    
    # Generate the multiple scattering LUT using the transmittance LUT
    print("Generating multiple scattering LUT...")
    multiscatt_gen.generate(command_encoder, transmittance_gen.get_texture())
    
    # Generate the sky view LUT using both previous LUTs
    print("Generating sky view LUT...")
    sky_view_gen.generate(
        command_encoder,
        transmittance_gen.get_texture(),
        multiscatt_gen.get_texture(),
        sun_direction=(0.0, 0.5, -0.866)  # Default: ~30 degrees elevation
    )
    
    # Submit commands
    device.submit_command_buffer(command_encoder.finish())
    
    print(f"\n[OK] Generated transmittance LUT: {transmittance_gen.LUT_WIDTH}x{transmittance_gen.LUT_HEIGHT}")
    print(f"     Texture: {transmittance_gen.get_texture()}")
    
    print(f"\n[OK] Generated multiple scattering LUT: {multiscatt_gen.LUT_WIDTH}x{multiscatt_gen.LUT_HEIGHT}")
    print(f"     Texture: {multiscatt_gen.get_texture()}")
    
    print(f"\n[OK] Generated sky view LUT: {sky_view_gen.LUT_WIDTH}x{sky_view_gen.LUT_HEIGHT}")
    print(f"     Texture: {sky_view_gen.get_texture()}")
    
    # Save all LUTs as images
    print("\nSaving transmittance_lut.png...")
    transmittance_gen.get_texture().to_bitmap().convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.uint8,
        srgb_gamma=True,
    ).write_async("transmittance_lut.png")
    
    print("Saving multiscatt_lut.png...")
    multiscatt_gen.get_texture().to_bitmap().convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.uint8,
        srgb_gamma=True,
    ).write_async("multiscatt_lut.png")
    
    print("Saving sky_view_lut.png...")
    sky_view_gen.get_texture().to_bitmap().convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.uint8,
        srgb_gamma=True,
    ).write_async("sky_view_lut.png")
    
    print("\nDone! All atmospheric LUTs generated successfully!")

