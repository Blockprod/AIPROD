"""
Tile blending utilities for seamless tiled inference.

Handles blending of overlapping tile borders to prevent visible seams
using various windowing functions (Gaussian, linear, cosine).
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange


class TileBlendingManager:
    """Manages blending of overlapping tiles."""
    
    @staticmethod
    def create_blend_mask_1d(
        size: int,
        overlap: int,
        window_type: str = "gaussian",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Create 1D blend mask for tile overlap region.
        
        Args:
            size: Total mask size
            overlap: Overlap region size
            window_type: Type of window ("gaussian", "linear", "cosine")
            device: Torch device
        
        Returns:
            1D mask tensor of shape (size,)
        """
        if device is None:
            device = torch.device("cpu")
        
        mask = torch.ones(size, dtype=torch.float32, device=device)
        
        if overlap <= 0:
            return mask
        
        if window_type == "linear":
            # Linear fade
            fade_in = torch.linspace(0, 1, overlap, device=device)
            fade_out = torch.linspace(1, 0, overlap, device=device)
            mask[:overlap] = fade_in
            mask[-overlap:] = fade_out
        
        elif window_type == "cosine":
            # Cosine fade (smoother)
            fade_in = (1 - torch.cos(torch.linspace(0, 3.14159, overlap, device=device))) / 2
            fade_out = (1 - torch.cos(torch.linspace(3.14159, 6.28318, overlap, device=device))) / 2
            mask[:overlap] = fade_in
            mask[-overlap:] = fade_out
        
        elif window_type == "gaussian":
            # Gaussian fade
            sigma = overlap / 3.0
            x = torch.arange(overlap, dtype=torch.float32, device=device)
            fade_in = torch.exp(-(x - overlap) ** 2 / (2 * sigma ** 2))
            fade_out = torch.exp(-(x) ** 2 / (2 * sigma ** 2))
            mask[:overlap] = fade_in
            mask[-overlap:] = fade_out
        
        return mask
    
    @staticmethod
    def create_blend_mask_2d(
        h: int,
        w: int,
        overlap_h: int,
        overlap_w: int,
        window_type: str = "gaussian",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Create 2D blend mask for spatial tile overlap.
        
        Args:
            h, w: Mask dimensions
            overlap_h, overlap_w: Overlap amounts
            window_type: Type of window
            device: Torch device
        
        Returns:
            2D mask tensor of shape (h, w)
        """
        if device is None:
            device = torch.device("cpu")
        
        # Create 1D masks
        mask_h = TileBlendingManager.create_blend_mask_1d(h, overlap_h, window_type, device)
        mask_w = TileBlendingManager.create_blend_mask_1d(w, overlap_w, window_type, device)
        
        # Create 2D mask via outer product
        mask_2d = rearrange(mask_h, "h -> h 1") * rearrange(mask_w, "w -> 1 w")
        
        return mask_2d
    
    @staticmethod
    def blend_spatial_tiles(
        output_tensor: torch.Tensor,
        tiles_dict: Dict[Tuple[int, int, int, int], torch.Tensor],
        overlap_h: int,
        overlap_w: int,
        window_type: str = "gaussian",
    ) -> torch.Tensor:
        """Blend spatial tiles (2D) with weighted averaging.
        
        Args:
            output_tensor: Output tensor to fill [B, C, H, W, T]
            tiles_dict: Dict mapping (h_start, h_end, w_start, w_end) -> tile tensor
            overlap_h, overlap_w: Overlap amounts
            window_type: Window function type
        
        Returns:
            Blended output tensor
        """
        device = output_tensor.device
        B, C, H, W, T = output_tensor.shape
        
        # Normalization tensor (weights sum)
        weight_sum = torch.zeros_like(output_tensor)
        
        # Fill output with weighted tiles
        for (h_start, h_end, w_start, w_end), tile in tiles_dict.items():
            tile_h = h_end - h_start
            tile_w = w_end - w_start
            
            # Create blend mask for this tile
            # Get actual overlap amounts for this specific tile
            actual_overlap_h = overlap_h if h_start > 0 else 0
            actual_overlap_w = overlap_w if w_start > 0 else 0
            
            mask = TileBlendingManager.create_blend_mask_2d(
                tile_h, tile_w,
                actual_overlap_h, actual_overlap_w,
                window_type, device
            )
            
            # Expand mask for batch and channel dims
            mask = rearrange(mask, "h w -> 1 1 h w 1")
            
            # Apply mask and add to output
            weighted_tile = tile * mask
            output_tensor[:, :, h_start:h_end, w_start:w_end, :] += weighted_tile
            weight_sum[:, :, h_start:h_end, w_start:w_end, :] += mask
        
        # Normalize by weight sum
        output_tensor = output_tensor / (weight_sum + 1e-8)
        
        return output_tensor
    
    @staticmethod
    def blend_temporal_tiles(
        frames_list: list[torch.Tensor],
        overlap_frames: int,
        window_type: str = "cosine",
    ) -> torch.Tensor:
        """Blend temporal tiles (frames) with smooth interpolation.
        
        Args:
            frames_list: List of frame tensors [B, C, H, W, F]
            overlap_frames: Number of overlapping frames
            window_type: Window function type
        
        Returns:
            Blended frames tensor [B, C, H, W, T_total]
        """
        if not frames_list:
            return torch.tensor([])
        
        if len(frames_list) == 1:
            return frames_list[0]
        
        device = frames_list[0].device
        B, C, H, W, _ = frames_list[0].shape
        
        # Allocate output
        # Total frames = sum(all_frames) - sum(overlaps)
        stride = frames_list[0].shape[-1] - overlap_frames
        total_frames = stride * (len(frames_list) - 1) + frames_list[-1].shape[-1]
        
        output = torch.zeros(B, C, H, W, total_frames, device=device, dtype=frames_list[0].dtype)
        weight_sum = torch.zeros(B, C, H, W, total_frames, device=device)
        
        # Process each chunk
        t_offset = 0
        for i, frames in enumerate(frames_list):
            F = frames.shape[-1]
            
            if i == 0:
                # First chunk - no blending at start
                output[:, :, :, :, :F] = frames
                weight_sum[:, :, :, :, :F] += 1.0
                t_offset = F
            else:
                # Blend overlap region
                if overlap_frames > 0 and i > 0:
                    # Get last overlap_frames from output and first overlap_frames from current
                    prev_overlap = output[:, :, :, :, t_offset - overlap_frames:t_offset]
                    curr_overlap = frames[:, :, :, :, :overlap_frames]
                    
                    # Create blend window
                    blend_window = TileBlendingManager.create_blend_mask_1d(
                        overlap_frames, overlap_frames, window_type, device
                    )
                    
                    # Blend frames
                    for j in range(overlap_frames):
                        blend_factor = blend_window[j].item()
                        blended = (1 - blend_factor) * prev_overlap[..., j] + blend_factor * curr_overlap[..., j]
                        output[:, :, :, :, t_offset - overlap_frames + j] = blended
                    
                    # Non-overlapping frames from current
                    output[:, :, :, :, t_offset:t_offset + F - overlap_frames] = frames[:, :, :, :, overlap_frames:]
                else:
                    # No blendingneed, just concatenate
                    output[:, :, :, :, t_offset:t_offset + F] = frames
                
                t_offset += F - overlap_frames
        
        return output
    
    @staticmethod
    def blend_hybrid_tiles(
        tiles_dict: Dict[Tuple[int, int, int, int, int, int], torch.Tensor],
        output_shape: Tuple[int, int, int, int, int],
        overlap_h: int,
        overlap_w: int,
        overlap_t: int,
        window_type: str = "gaussian",
    ) -> torch.Tensor:
        """Blend hybrid (spatial + temporal) tiles.
        
        Args:
            tiles_dict: Dict mapping (h_s, h_e, w_s, w_e, t_s, t_e) -> tile tensor
            output_shape: Output shape (B, C, H, W, T)
            overlap_h, overlap_w, overlap_t: Overlap amounts
            window_type: Window function type
        
        Returns:
            Blended output tensor
        """
        device = next(iter(tiles_dict.values())).device
        B, C, H, W, T = output_shape
        
        output = torch.zeros(B, C, H, W, T, device=device, dtype=next(iter(tiles_dict.values())).dtype)
        weight_sum = torch.zeros(B, C, H, W, T, device=device)
        
        # Create 3D mask pattern
        for (h_s, h_e, w_s, w_e, t_s, t_e), tile in tiles_dict.items():
            tile_h, tile_w, tile_t = h_e - h_s, w_e - w_s, t_e - t_s
            
            # Blend factors
            actual_overlap_h = overlap_h if h_s > 0 else 0
            actual_overlap_w = overlap_w if w_s > 0 else 0
            actual_overlap_t = overlap_t if t_s > 0 else 0
            
            # Create 3D mask
            mask_h = TileBlendingManager.create_blend_mask_1d(tile_h, actual_overlap_h, window_type, device)
            mask_w = TileBlendingManager.create_blend_mask_1d(tile_w, actual_overlap_w, window_type, device)
            mask_t = TileBlendingManager.create_blend_mask_1d(tile_t, actual_overlap_t, window_type, device)
            
            # Combine into 3D
            mask_3d = rearrange(mask_h, "h -> h 1 1") * rearrange(mask_w, "w -> 1 w 1") * rearrange(mask_t, "t -> 1 1 t")
            mask_3d = rearrange(mask_3d, "h w t -> 1 1 h w t")
            
            # Apply and accumulate
            weighted_tile = tile * mask_3d
            output[:, :, h_s:h_e, w_s:w_e, t_s:t_e] += weighted_tile
            weight_sum[:, :, h_s:h_e, w_s:w_e, t_s:t_e] += mask_3d
        
        # Normalize
        output = output / (weight_sum + 1e-8)
        
        return output
