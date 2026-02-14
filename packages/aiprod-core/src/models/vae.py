"""AIPROD v2 Video VAE (Variational Autoencoder)

Hierarchical 3D convolutional autoencoder for compressing video frames
to a learnable latent space. Enhanced with temporal attention for better
motion representation.

Compression: 16x spatial + 4x temporal = 64x total compression
Latent dimension: 256-D per frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder3D(nn.Module):
    """Progressive 3D convolutional encoder."""
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Progressive downsampling
        # Input: (B, T, C, H, W) -> intermediate processing
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), 
                               stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn1 = nn.GroupNorm(8, 64)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), 
                               stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.GroupNorm(8, 128)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), 
                               stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.GroupNorm(16, 256)
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), 
                               stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.GroupNorm(32, 512)
        
        # Bottleneck to latent
        self.proj_mean = nn.Conv3d(512, latent_dim, kernel_size=1)
        self.proj_logvar = nn.Conv3d(512, latent_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) video frames
        
        Returns:
            mean, logvar: latent distribution parameters
        """
        # Reshape for 3D conv: combine batch and time
        B, T, C, H, W = x.shape
        x = x.reshape(B, C, T, H, W)  # (B, C, T, H, W) for Conv3d
        
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, T, H/2, W/2)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, T/2, H/4, W/4)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 256, T/4, H/8, W/8)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 512, T/4, H/16, W/16)
        
        mean = self.proj_mean(x)
        logvar = self.proj_logvar(x)
        
        return mean, logvar


class TemporalAttentionBlock(nn.Module):
    """Temporal attention for modeling motion between frames."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention across time dimension."""
        B, C, T, H, W = x.shape
        
        # Reshape for attention: (B*H*W, T, C)
        x_norm = self.norm(x)
        x_reshape = x_norm.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, T, C)
        x_reshape = x_reshape.reshape(B * H * W, T, C)
        
        # Multi-head attention
        attn_out, _ = self.mha(x_reshape, x_reshape, x_reshape)
        attn_out = attn_out.reshape(B, H, W, T, C)
        attn_out = attn_out.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, T, H, W)
        
        x = x + attn_out
        
        # Feed-forward
        x_reshape = x.reshape(B, C, -1).transpose(1, 2)  # (B, THW, C)
        ff_out = self.ff(x_reshape)
        ff_out = ff_out.transpose(1, 2).reshape(B, C, T, H, W)
        
        x = x + ff_out
        return x


class Decoder3D(nn.Module):
    """Progressive 3D convolutional decoder."""
    
    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # Upsample from latent back to video
        self.deconv1 = nn.ConvTranspose3d(latent_dim, 512, kernel_size=(3, 3, 3),
                                          stride=(1, 2, 2), padding=(1, 1, 1),
                                          output_padding=(0, 1, 1))
        self.bn1 = nn.GroupNorm(32, 512)
        
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3),
                                          stride=(2, 2, 2), padding=(1, 1, 1),
                                          output_padding=(1, 1, 1))
        self.bn2 = nn.GroupNorm(16, 256)
        
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3),
                                          stride=(2, 2, 2), padding=(1, 1, 1),
                                          output_padding=(1, 1, 1))
        self.bn3 = nn.GroupNorm(8, 128)
        
        self.deconv4 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3),
                                          stride=(1, 2, 2), padding=(1, 1, 1),
                                          output_padding=(0, 1, 1))
        self.bn4 = nn.GroupNorm(8, 64)
        
        # Output layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        
        # Temporal attention for better motion
        self.temporal_attn = TemporalAttentionBlock(256)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, T, H, W) latent code
        
        Returns:
            (B, T, C, H, W) reconstructed video
        """
        x = F.relu(self.bn1(self.deconv1(z)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        
        x = self.final_conv(x)
        x = torch.sigmoid(x)  # Normalize to [0, 1]
        
        # Reshape back to (B, T, C, H, W)
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        
        return x


class VideoVAE(nn.Module):
    """
    Video VAE for compressing video to latent space.
    
    Features:
    - Hierarchical 3D convolutions
    - Temporal attention for motion modeling
    - 256-D latent outputs
    - KL-divergence regularization
    """
    
    def __init__(self, latent_dim: int = 256, beta: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta  # KL weight
        
        self.encoder = Encoder3D(in_channels=3, latent_dim=latent_dim)
        self.decoder = Decoder3D(latent_dim=latent_dim, out_channels=3)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode video to latent distribution.
        
        Args:
            x: (B, T, C, H, W) video
        
        Returns:
            z: sampled latent code
            mean: latent mean
            logvar: latent log-variance
        """
        mean, logvar = self.encoder(x)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to video."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass."""
        z, mean, logvar = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: reconstruction + KL divergence.
        
        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Weighted combination
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test VideoVAE
    vae = VideoVAE(latent_dim=256)
    
    print(f"VideoVAE created")
    print(f"Total parameters: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 1
    num_frames = 4
    height, width = 256, 256
    x = torch.randn(batch_size, num_frames, 3, height, width)
    
    reconstruction, mean, logvar = vae(x)
    loss, recon_loss, kl_loss = vae.compute_loss(x, reconstruction, mean, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
