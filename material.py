import slangpy as spy
from typing import Optional


# Alpha modes (matches glTF spec)
ALPHA_MODE_OPAQUE = 0  # fully opaque
ALPHA_MODE_MASK = 1    # discard if alpha < cutoff
ALPHA_MODE_BLEND = 2   # stochastic alpha


class Material:
    """
    PBR Material class
    
    Supports constant values and texture paths:
    - base_color / base_color_texture: base color
    - emissive / emissive_texture: emissive
    - roughness / roughness_texture: roughness
    - metallic / metallic_texture: metallic
    - normal_texture: normal map
    - alpha_mode: Alpha mode (OPAQUE/MASK/BLEND)
    - alpha_cutoff: Alpha cutoff threshold (default 0.5)
    """
    
    def __init__(self, 
                 base_color: "spy.float3param" = spy.float3(0.5),
                 emissive: "spy.float3param" = spy.float3(0.0),
                 roughness: float = 0.5,
                 metallic: float = 0.0,
                 # PBR texture paths (None means use constant value)
                 base_color_texture: Optional[str] = None,
                 normal_texture: Optional[str] = None,
                 roughness_texture: Optional[str] = None,
                 metallic_texture: Optional[str] = None,
                 emissive_texture: Optional[str] = None,
                 # Alpha related properties
                 alpha_mode: int = ALPHA_MODE_OPAQUE,
                 alpha_cutoff: float = 0.5):
        super().__init__()
        # Constant values
        self.base_color = base_color
        self.emissive = emissive
        self.roughness = roughness
        self.metallic = metallic
        
        # Texture paths
        self.base_color_texture = base_color_texture
        self.normal_texture = normal_texture
        self.roughness_texture = roughness_texture
        self.metallic_texture = metallic_texture
        self.emissive_texture = emissive_texture
        
        # Alpha related
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff
        
        # TODO: implement later
        self.flags = 65535
        self.shading_model = 65535
    
    def has_any_texture(self) -> bool:
        """Check if any texture is present"""
        return any([
            self.base_color_texture,
            self.normal_texture,
            self.roughness_texture,
            self.metallic_texture,
            self.emissive_texture
        ])



