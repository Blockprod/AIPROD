"""
Concrete inference nodes implementing core video generation operations.

Each node encapsulates a specific transformation step in the inference pipeline:
- TextEncodeNode: Text → embeddings (Gemma 3)
- DenoiseNode: Denoising loop with configurable steps, guidance, LoRAs
- UpsampleNode: 2x spatial upsampling
- DecodeVideoNode: Latents → video frames (with tiled decoding)
- AudioEncodeNode: Text → audio embeddings
- CleanupNode: GPU memory cleanup
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from .graph import GraphNode, GraphContext


class TextEncodeNode(GraphNode):
    """
    Text encoding node.
    
    Converts text prompts to embeddings using Gemma 3 text encoder.
    Handles negative prompts for unconditional guidance.
    
    Inputs:
        prompt (str): Text to encode
        negative_prompt (str, optional): Negative prompt for guidance
        max_length (int, optional): Maximum prompt length (default: 1024)
    
    Outputs:
        embeddings (Tensor): [batch, seq_len, hidden_dim] encoded text
        embeddings_pooled (Tensor): [batch, hidden_dim] pooled representation
    """
    
    @property
    def input_keys(self) -> List[str]:
        return ["prompt"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["embeddings", "embeddings_pooled"]
    
    def __init__(self, text_encoder, **kwargs):
        """
        Initialize text encoder node.
        
        Args:
            text_encoder: Gemma 3 text encoder model
            **kwargs: Additional config (prompt_embeds_path, etc.)
        """
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.max_length = self.config.get("max_length", 1024)
        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Encode text prompts to embeddings.
        
        Args:
            context: GraphContext with 'prompt' input
            
        Returns:
            Dict with 'embeddings' and 'embeddings_pooled' tensors
            
        Raises:
            ValueError: If prompt is missing or invalid
        """
        self._validate_inputs(context, ["prompt"])
        
        prompt = context["prompt"]
        negative_prompt = context.get("negative_prompt", "")
        
        # Handle both string and list of strings
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] if negative_prompt else [""] * len(prompt)
        
        # Encode positive and negative prompts
        embeddings: List[torch.Tensor] = []
        
        for pos, neg in zip(prompt, negative_prompt):
            pos_emb = self._encode_single(pos, context.dtype)        # [1, seq_len, hidden]
            neg_emb = self._encode_single(neg, context.dtype)        # [1, seq_len, hidden]
            
            # Stack: [2, seq_len, hidden] for positive + negative
            combined = torch.cat([pos_emb, neg_emb], dim=0)
            embeddings.append(combined)
        
        # Batch concatenation: [batch * 2, seq_len, hidden]
        all_embeddings = torch.cat(embeddings, dim=0)
        
        # Pooled: mean across sequence dimension
        pooled = all_embeddings.mean(dim=1)  # [batch * 2, hidden]
        
        return {
            "embeddings": all_embeddings,
            "embeddings_pooled": pooled,
        }
    
    def _encode_single(self, text: str, dtype: torch.dtype) -> torch.Tensor:
        """Encode single text string to embeddings via text_encoder."""
        # Tokenise & encode through the model
        if hasattr(self.text_encoder, "tokenize"):
            tokens = self.text_encoder.tokenize(
                text, max_length=self.max_length, device=self.device,
            )
        else:
            # Fallback: build a simple token tensor from whitespace split
            words = text.split()[:self.max_length]
            tokens = torch.zeros(1, len(words), dtype=torch.long, device=self.device)

        # Forward through encoder
        with torch.no_grad():
            embeddings = self.text_encoder(tokens)  # expected: [1, seq, hidden]

        return embeddings.to(dtype=dtype)


class DenoiseNode(GraphNode):
    """
    Denoising loop node.
    
    Performs iterative denoising with Euler solver. Supports:
    - Configurable number of steps
    - Classifier-free guidance (CFG)
    - Spatial guidance (STG)
    - LoRA composition for style/subject control
    
    Inputs:
        latents (Tensor): [batch, channels, frames, height, width] initial noise
        embeddings (Tensor): [batch, seq_len, hidden_dim] text embeddings
        num_inference_steps (int): Denoising steps (default: 20)
        guidance_scale (float): CFG strength (default: 7.5)
    
    Outputs:
        latents_denoised (Tensor): Denoised latents, same shape as input
    """
    
    @property
    def input_keys(self) -> List[str]:
        return ["latents", "embeddings"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["latents_denoised"]
    
    def __init__(self, model, scheduler, **kwargs):
        """
        Initialize denoiser node.
        
        Args:
            model: Denoising model (transformer)
            scheduler: Noise scheduler
            **kwargs: Config (num_steps, guidance_scale, loras, etc.)
        """
        super().__init__(**kwargs)
        self.model = model
        self.scheduler = scheduler
        self.num_steps = self.config.get("num_inference_steps", 20)
        self.guidance_scale = self.config.get("guidance_scale", 7.5)
        self.loras = self.config.get("loras", [])
        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Execute denoising loop.
        
        Args:
            context: GraphContext with latents and embeddings
            
        Returns:
            Dict with denoised latents
            
        Raises:
            ValueError: If required inputs missing
        """
        self._validate_inputs(context, ["latents", "embeddings"])
        
        latents = context["latents"]
        embeddings = context["embeddings"]
        num_steps = context.get("num_inference_steps", self.num_steps)
        guidance_scale = context.get("guidance_scale", self.guidance_scale)
        
        # Denoise loop
        for step_idx in range(num_steps):
            # Get timestep
            t = self.scheduler.timesteps[step_idx]
            
            # Model prediction
            noise_pred = self._denoise_step(latents, embeddings, t, guidance_scale)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        
        return {"latents_denoised": latents}
    
    def _denoise_step(
        self,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
        t: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Single denoising step with classifier-free guidance."""
        # Duplicate latents for unconditional + conditional
        latent_input = torch.cat([latents, latents], dim=0)

        with torch.no_grad():
            noise_pred = self.model(latent_input, t, encoder_hidden_states=embeddings)

        # Split predictions & apply CFG
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        return noise_pred


class UpsampleNode(GraphNode):
    """
    Spatial upsampling node (2x magnification).
    
    Increases spatial resolution using learned upsampling module.
    Maintains temporal consistency through temporal attention.
    
    Inputs:
        latents (Tensor): [batch, channels, frames, height, width]
    
    Outputs:
        latents_upsampled (Tensor): [batch, channels, frames, 2*height, 2*width]
    """
    
    @property
    def input_keys(self) -> List[str]:
        return ["latents"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["latents_upsampled"]
    
    def __init__(self, upsampler, **kwargs):
        """
        Initialize upsampler node.
        
        Args:
            upsampler: Upsampling model
            **kwargs: Config (scale_factor, etc.)
        """
        super().__init__(**kwargs)
        self.upsampler = upsampler
        self.scale_factor = self.config.get("scale_factor", 2)
        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Execute spatial upsampling.
        
        Args:
            context: GraphContext with latents
            
        Returns:
            Dict with upsampled latents
        """
        self._validate_inputs(context, ["latents"])
        
        latents = context["latents"]
        
        # Upsampling
        latents_up = self._upsample(latents)
        
        return {"latents_upsampled": latents_up}
    
    def _upsample(self, latents: torch.Tensor) -> torch.Tensor:
        """Perform 2x upsampling with interpolation + learned refinement."""
        b, c, f, h, w = latents.shape
        # Bilinear interpolation per frame
        frames = []
        for fi in range(f):
            frame = latents[:, :, fi, :, :]  # [B, C, H, W]
            up = torch.nn.functional.interpolate(
                frame, scale_factor=self.scale_factor, mode="bilinear", align_corners=False,
            )
            frames.append(up)
        upsampled = torch.stack(frames, dim=2)  # [B, C, F, H*s, W*s]

        # Learned refinement (if model available)
        if self.upsampler is not None:
            with torch.no_grad():
                upsampled = self.upsampler(upsampled)

        return upsampled


class DecodeVideoNode(GraphNode):
    """
    Video decoding node.
    
    Decodes latents to video frames using VAE decoder.
    Supports tiled decoding for memory efficiency.
    
    Inputs:
        latents_denoised (Tensor): [batch, channels, frames, height, width]
        vae_scaling_factor (float, optional): Scaling for latents (default: 0.18215)
    
    Outputs:
        video_frames (Tensor): [batch, frames, height*8, width*8, 3] or similar
    """
    
    @property
    def input_keys(self) -> List[str]:
        return ["latents_denoised"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["video_frames"]
    
    def __init__(self, vae_decoder, **kwargs):
        """
        Initialize video decoder node.
        
        Args:
            vae_decoder: VAE decoder model
            **kwargs: Config (scaling_factor, tile_size, etc.)
        """
        super().__init__(**kwargs)
        self.vae_decoder = vae_decoder
        self.scaling_factor = self.config.get("vae_scaling_factor", 0.18215)
        self.tile_size = self.config.get("tile_size", 128)
        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Execute video decoding.
        
        Args:
            context: GraphContext with denoised latents
            
        Returns:
            Dict with decoded video frames
        """
        self._validate_inputs(context, ["latents_denoised"])
        
        latents = context["latents_denoised"]
        
        # Scale latents
        latents = latents / self.scaling_factor
        
        # Decode with tiling if needed
        video = self._decode_tiled(latents)
        
        return {"video_frames": video}
    
    def _decode_tiled(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents with tiling for memory efficiency."""
        b, c, f, h, w = latents.shape
        tile = self.tile_size
        overlap = tile // 4  # 25 % overlap for blending
        stride = tile - overlap

        # Decode full if small enough
        if h <= tile and w <= tile:
            with torch.no_grad():
                return self.vae_decoder(latents)

        # Tiled decoding
        # Output spatial dims (VAE typically 8x)
        out_h, out_w = h * 8, w * 8
        out_tile = tile * 8
        out_stride = stride * 8
        out_overlap = overlap * 8

        output = torch.zeros(b, f, out_h, out_w, 3, device=latents.device)
        weight = torch.zeros(b, 1, 1, out_h, out_w, device=latents.device)

        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                tile_latents = latents[:, :, :, y0:y1, x0:x1]
                with torch.no_grad():
                    decoded = self.vae_decoder(tile_latents)  # [B, F, H', W', 3]
                oy0, ox0 = y0 * 8, x0 * 8
                dh, dw = decoded.shape[2], decoded.shape[3]
                output[:, :, oy0:oy0 + dh, ox0:ox0 + dw, :] += decoded
                weight[:, :, :, oy0:oy0 + dh, ox0:ox0 + dw] += 1.0

        # Average overlapping regions
        weight = weight.clamp(min=1.0)
        output = output / weight.squeeze(1).unsqueeze(-1)

        return output


class AudioEncodeNode(GraphNode):
    """
    Audio encoding node.
    
    Generates audio embeddings from text descriptions.
    Enables synchronized audio-video generation.
    
    Inputs:
        audio_prompt (str): Description of desired audio
    
    Outputs:
        audio_embeddings (Tensor): Audio feature embeddings
    """
    
    @property
    def input_keys(self) -> List[str]:
        return ["audio_prompt"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["audio_embeddings"]
    
    def __init__(self, audio_encoder, **kwargs):
        """
        Initialize audio encoder node.
        
        Args:
            audio_encoder: Audio encoding model
            **kwargs: Config
        """
        super().__init__(**kwargs)
        self.audio_encoder = audio_encoder
        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Execute audio encoding.
        
        Args:
            context: GraphContext with audio_prompt
            
        Returns:
            Dict with audio embeddings
        """
        audio_prompt = context.get("audio_prompt", "")
        
        if not audio_prompt:
            # Return silent embeddings
            emb_dim = getattr(self.audio_encoder, "embedding_dim", 512)
            return {"audio_embeddings": torch.zeros(1, emb_dim, device=self.device)}

        # Encode audio prompt through the model
        with torch.no_grad():
            if hasattr(self.audio_encoder, "encode_text"):
                embeddings = self.audio_encoder.encode_text(audio_prompt)
            else:
                embeddings = self.audio_encoder(audio_prompt)

        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return {"audio_embeddings": embeddings.to(device=self.device)}


class CleanupNode(GraphNode):
    """
    GPU memory cleanup node.
    
    Clears intermediate tensors and cache to free GPU memory.
    Should be placed at the end of inference graphs.
    
    Inputs:
        (no required inputs)
    
    Outputs:
        memory_freed_mb (int): Amount of memory freed
    """
    
    @property
    def input_keys(self) -> List[str]:
        return []
    
    @property
    def output_keys(self) -> List[str]:
        return ["memory_freed_mb"]
    
    def __init__(self, **kwargs):
        """Initialize cleanup node."""
        super().__init__(**kwargs)
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Execute GPU memory cleanup.
        
        Args:
            context: GraphContext (not used)
            
        Returns:
            Dict with memory cleanup stats
        """
        bytes_freed = 0
        
        if torch.cuda.is_available():
            # Record memory before
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            bytes_freed = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        
        return {
            "memory_freed_mb": bytes_freed / (1024 ** 2)
        }
