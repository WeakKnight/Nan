"""Texture Manager - Path-based cached texture loading and Bindless Handle management"""
import slangpy as spy
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum

# Global debug switch
texture_loader_debug_print = False


class TextureType(IntEnum):
    """Texture type, used to determine loading options and default values"""
    BASE_COLOR = 0       # sRGB, default white
    NORMAL = 1           # Linear, default (0.5, 0.5, 1.0, 1.0)
    ROUGHNESS = 2        # Linear, default 0.5
    METALLIC = 3         # Linear, default 0.0
    EMISSIVE = 4         # sRGB, default black
    SPECULAR_COLOR = 5   # sRGB, default white


@dataclass
class TextureRecord:
    """Texture cache record"""
    texture: spy.Texture       # GPU texture object
    view: spy.TextureView      # Texture view (for getting Bindless Handle)
    handle: int                # Bindless descriptor handle
    path: str                  # Original file path (used as cache key)


class TextureManager:
    """
    Texture Manager
    
    Features:
    - Load textures from files, auto-cache same paths
    - Auto-generate MIP chain
    - Manage Bindless Texture Handle
    - Provide default textures (1x1 pixel)
    """
    
    # Default texture colors (RGBA, 0-255)
    DEFAULT_COLORS = {
        TextureType.BASE_COLOR: [255, 255, 255, 255],    # white
        TextureType.NORMAL: [128, 128, 255, 255],        # normal blue (0.5, 0.5, 1.0)
        TextureType.ROUGHNESS: [128, 128, 128, 255],     # 0.5 gray
        TextureType.METALLIC: [0, 0, 0, 255],            # black (non-metallic)
        TextureType.EMISSIVE: [0, 0, 0, 255],            # black (no emission)
        TextureType.SPECULAR_COLOR: [255, 255, 255, 255], # white (default specular)
    }
    
    def __init__(self, device: spy.Device):
        """
        Initialize texture manager
        
        Args:
            device: SlangPy Device
        """
        self.device = device
        self.loader = spy.TextureLoader(device)
        self._texture_cache: Dict[str, TextureRecord] = {}
        self._default_textures: Dict[TextureType, TextureRecord] = {}
        self._create_default_textures()
        
        if texture_loader_debug_print:
            print(f"[TextureManager] Initialized with device: {device}")
            print(f"[TextureManager] Created {len(self._default_textures)} default textures")
    
    def load_texture(self, 
                     path: str, 
                     texture_type: TextureType = TextureType.BASE_COLOR,
                     generate_mips: bool = True) -> TextureRecord:
        """
        Load texture, return from cache if hit
        
        Args:
            path: Texture file path
            texture_type: Texture type, determines whether to use sRGB
            generate_mips: Whether to generate MIP chain
            
        Returns:
            TextureRecord: Contains texture, view and Bindless Handle
        """
        # Normalize path as cache key
        cache_key = str(Path(path).resolve())
        
        # Cache hit
        if cache_key in self._texture_cache:
            if texture_loader_debug_print:
                print(f"[TextureManager] Cache hit: {cache_key}")
            return self._texture_cache[cache_key]
        
        # Determine whether to use sRGB
        is_srgb = texture_type in (TextureType.BASE_COLOR, TextureType.EMISSIVE, TextureType.SPECULAR_COLOR)
        
        # Try alternative extensions if file doesn't exist
        actual_path = path
        if not Path(path).exists():
            # Common texture format alternatives
            alternatives = ['.dds', '.png', '.jpg', '.jpeg', '.tga', '.bmp', '.exr']
            stem = Path(path).stem
            parent = Path(path).parent
            for alt_ext in alternatives:
                alt_path = parent / (stem + alt_ext)
                if alt_path.exists():
                    actual_path = str(alt_path)
                    if texture_loader_debug_print:
                        print(f"[TextureManager] File not found: {path}")
                        print(f"[TextureManager]   Using alternative: {actual_path}")
                    break
        
        if texture_loader_debug_print:
            print(f"[TextureManager] Loading texture: {actual_path}")
            print(f"[TextureManager]   Type: {texture_type.name}")
            print(f"[TextureManager]   sRGB: {is_srgb}")
            print(f"[TextureManager]   Generate MIPs: {generate_mips}")
        
        # Load texture (with error handling)
        try:
            texture = self.loader.load_texture(
                actual_path,
                options={
                    "generate_mips": generate_mips,
                    "load_as_srgb": is_srgb,
                }
            )
        except Exception as e:
            print(f"[TextureManager] ERROR: Failed to load texture: {actual_path}")
            print(f"[TextureManager]   Error: {e}")
            print(f"[TextureManager]   Using default texture for type: {texture_type.name}")
            # Return default texture on failure
            default_record = self._default_textures[texture_type]
            # Cache with original path to avoid repeated load attempts
            self._texture_cache[cache_key] = default_record
            return default_record
        
        # Create view and get Bindless Handle
        view = texture.create_view()
        handle = view.descriptor_handle_ro.value  # Extract integer value from DescriptorHandle
        
        if texture_loader_debug_print:
            print(f"[TextureManager]   Size: {texture.width}x{texture.height}")
            print(f"[TextureManager]   Format: {texture.format}")
            print(f"[TextureManager]   MIP levels: {texture.mip_count}")
            print(f"[TextureManager]   Bindless handle: 0x{handle:08x}")
        
        # Create record and cache
        record = TextureRecord(
            texture=texture,
            view=view,
            handle=handle,
            path=cache_key
        )
        self._texture_cache[cache_key] = record
        
        if texture_loader_debug_print:
            print(f"[TextureManager]   Cached. Total cached: {len(self._texture_cache)}")
        
        return record
    
    def get_default_texture(self, texture_type: TextureType) -> TextureRecord:
        """
        Get default texture (1x1 pixel)
        
        Args:
            texture_type: Texture type
            
        Returns:
            TextureRecord: Default texture record
        """
        if texture_loader_debug_print:
            print(f"[TextureManager] Using default texture for: {texture_type.name}")
        return self._default_textures[texture_type]
    
    def get_or_default(self, 
                       path: Optional[str], 
                       texture_type: TextureType) -> TextureRecord:
        """
        Load texture, return default if path is None
        
        Args:
            path: Texture path, None means use default
            texture_type: Texture type
            
        Returns:
            TextureRecord: Texture record
        """
        if path is None:
            return self.get_default_texture(texture_type)
        return self.load_texture(path, texture_type)
    
    def clear_cache(self) -> None:
        """Clear texture cache (excluding default textures)"""
        if texture_loader_debug_print:
            print(f"[TextureManager] Clearing cache. Removing {len(self._texture_cache)} textures")
        self._texture_cache.clear()
    
    def _create_default_textures(self) -> None:
        """Create default textures (1x1 pixel)"""
        for tex_type in TextureType:
            color = self.DEFAULT_COLORS[tex_type]
            is_srgb = tex_type in (TextureType.BASE_COLOR, TextureType.EMISSIVE, TextureType.SPECULAR_COLOR)
            
            # Create 1x1 RGBA data
            data = np.array([[color]], dtype=np.uint8)
            
            # Select format
            fmt = spy.Format.rgba8_unorm_srgb if is_srgb else spy.Format.rgba8_unorm
            
            # Create texture
            texture = self.device.create_texture(
                width=1,
                height=1,
                format=fmt,
                usage=spy.TextureUsage.shader_resource,
                data=data,
            )
            
            # Create view and get Handle
            view = texture.create_view()
            handle = view.descriptor_handle_ro.value  # Extract integer value
            
            # Store default texture
            self._default_textures[tex_type] = TextureRecord(
                texture=texture,
                view=view,
                handle=handle,
                path=f"__default_{tex_type.name}__"
            )
            
            if texture_loader_debug_print:
                print(f"[TextureManager] Created default texture: {tex_type.name}")
                print(f"[TextureManager]   Color: {color}")
                print(f"[TextureManager]   Format: {fmt}")
                print(f"[TextureManager]   Handle: 0x{handle:08x}")
    
    @property
    def cache_size(self) -> int:
        """Return current number of cached textures"""
        return len(self._texture_cache)
