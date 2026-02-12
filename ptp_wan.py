# Copyright 2024 Google LLC & Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prompt-to-Prompt for Wan2.1 Video Generation Model.

This module provides attention control classes for manipulating attention
in the Wan2.1 DiT-based video generation model.

Features:
- AttentionStore: Store and aggregate attention maps
- AttentionReplace: Word swap via attention replacement
- AttentionRefine: Attention refinement for prompt edits
- AttentionReweight: Reweight attention for emphasis control
- LocalBlend: Spatial-temporal blending for localized edits
"""

from typing import Optional, Union, Tuple, List, Dict, Callable
import torch
import torch.nn.functional as F
import numpy as np
import abc
from PIL import Image
import matplotlib.pyplot as plt

import ptp_utils_wan
import seq_aligner


# ============================================================================
# Configuration
# ============================================================================
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 5.0
MAX_NUM_WORDS = 512 # T5 max length (vs 77 for CLIP)
NUM_TRANSFORMER_BLOCKS = 30  # Wan2.1-1.3B has 30 blocks


# ============================================================================
# Model Loading
# ============================================================================

def load_wan_model(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    enable_offload: bool = False,
):
    """
    Load Wan2.1 video generation model.

    Args:
        model_id: HuggingFace model ID
        device: Target device
        torch_dtype: Data type for model weights
        enable_offload: When True, keep only the transformer on GPU and
                        move text_encoder + VAE to CPU immediately after
                        loading.  This dramatically reduces idle VRAM usage.

    Returns:
        Tuple of (pipeline, tokenizer, device)
    """
    from diffusers import WanPipeline, AutoencoderKLWan

    if device is None:
        device = ptp_utils_wan.get_device()

    print(f"Loading Wan2.1 model from {model_id}...")

    # Load VAE with float32 for stability
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # Load pipeline
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch_dtype,
    ).to(device)

    tokenizer = pipe.tokenizer

    print(f"Model loaded on {device}")
    print(f"Transformer blocks: {len(pipe.transformer.blocks)}")

    if enable_offload:
        pipe.text_encoder.to("cpu")
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        print("Offload enabled: text_encoder and VAE moved to CPU")

    return pipe, tokenizer, device


# ============================================================================
# Base Attention Controller
# ============================================================================

class AttentionControlWan(abc.ABC):
    """
    Base class for attention control in Wan2.1.

    This replaces the original AttentionControl class with adaptations
    for DiT architecture (flat transformer blocks instead of U-Net).
    """

    def step_callback(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Called after each diffusion step.

        Args:
            x_t: Current latent tensor [B, C, T, H, W]

        Returns:
            Modified latent tensor
        """
        return x_t

    def between_steps(self):
        """Called between diffusion steps for bookkeeping."""
        return

    @property
    def num_uncond_att_layers(self) -> int:
        """Number of unconditional attention layers (for low resource mode)."""
        return self.num_att_layers if self.low_resource else 0

    @abc.abstractmethod
    def forward(self, attn: torch.Tensor, is_cross: bool, block_idx: int) -> torch.Tensor:
        """
        Process attention weights.

        Args:
            attn: Attention weights [B*heads, seq_q, seq_kv]
            is_cross: True for cross-attention, False for self-attention
            block_idx: Transformer block index (0 to NUM_TRANSFORMER_BLOCKS-1)

        Returns:
            Modified attention weights
        """
        raise NotImplementedError

    def __call__(self, attn: torch.Tensor, is_cross: bool, block_idx: int) -> torch.Tensor:
        """
        Main entry point for attention manipulation.

        Handles unconditional/conditional split and layer counting.
        """
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, block_idx)
            else:
                # Split batch: first half is unconditional, second half is conditional
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, block_idx)

        self.cur_att_layer += 1

        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        return attn

    def reset(self):
        """Reset step and layer counters."""
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, low_resource: bool = False):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.low_resource = low_resource


class EmptyControlWan(AttentionControlWan):
    """Empty controller that passes attention through unchanged."""

    def forward(self, attn: torch.Tensor, is_cross: bool, block_idx: int) -> torch.Tensor:
        return attn


# ============================================================================
# Attention Store
# ============================================================================

class AttentionStoreWan(AttentionControlWan):
    """
    Store attention maps from Wan2.1 transformer blocks.

    Unlike U-Net which has down/mid/up structure, DiT has flat blocks.
    We store attention maps indexed by block number.
    """

    @staticmethod
    def get_empty_store() -> Dict[str, List]:
        """
        Create empty attention store.

        Structure:
            - blocks_cross: List of cross-attention maps from all blocks
            - blocks_self: List of self-attention maps from all blocks
        """
        return {
            "blocks_cross": [],
            "blocks_self": [],
        }

    def forward(self, attn: torch.Tensor, is_cross: bool, block_idx: int) -> torch.Tensor:
        """
        Store attention map and return unchanged.

        Args:
            attn: Attention weights [B*heads, seq_q, seq_kv]
            is_cross: Whether this is cross-attention
            block_idx: Which transformer block
        """
        key = "blocks_cross" if is_cross else "blocks_self"

        # Filter by size to avoid memory issues
        # For video, attention maps can be very large
        if attn.shape[1] <= self.max_store_size:
            # Store on CPU to save GPU memory
            self.step_store[key].append(attn.cpu().detach())

        return attn

    def between_steps(self):
        """Accumulate attention maps across steps."""
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]

        self.step_store = self.get_empty_store()

    def get_average_attention(self) -> Dict[str, List[torch.Tensor]]:
        """Get attention maps averaged over all steps."""
        return {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }

    def reset(self):
        super().reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, low_resource: bool = False, max_store_size: int = 4096):
        """
        Initialize attention store.

        Args:
            low_resource: Whether to use low resource mode
            max_store_size: Maximum sequence length to store (to avoid OOM)
        """
        super().__init__(low_resource)
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.max_store_size = max_store_size


# ============================================================================
# Attention Edit Controllers
# ============================================================================

class AttentionControlEditWan(AttentionStoreWan, abc.ABC):
    """
    Base class for attention editing operations.

    Extends AttentionStore to support attention replacement/refinement.
    """

    def step_callback(self, x_t: torch.Tensor) -> torch.Tensor:
        """Apply local blend if configured."""
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(
        self,
        attn_base: torch.Tensor,
        attn_replace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace self-attention maps.

        Args:
            attn_base: Base attention from source prompt
            attn_replace: Attention to replace

        Returns:
            Replaced attention
        """
        # For small spatial resolutions, use base attention
        if attn_replace.shape[1] <= 32 ** 2:
            return attn_base.unsqueeze(0).expand(attn_replace.shape[0], *attn_base.shape)
        return attn_replace

    @abc.abstractmethod
    def replace_cross_attention(
        self,
        attn_base: torch.Tensor,
        attn_replace: torch.Tensor,
    ) -> torch.Tensor:
        """Replace cross-attention maps. Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, attn: torch.Tensor, is_cross: bool, block_idx: int) -> torch.Tensor:
        """
        Apply attention editing.

        Args:
            attn: Attention weights [B*heads, seq_q, seq_kv]
            is_cross: Whether cross-attention
            block_idx: Transformer block index
        """
        # First, store the attention
        super().forward(attn, is_cross, block_idx)

        # Check if we should apply replacement at this step
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            # Reshape to separate batch dimension
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])

            attn_base, attn_replace = attn[0], attn[1:]

            if is_cross: # CA replacement
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = (
                    self.replace_cross_attention(attn_base, attn_replace) * alpha_words +
                    (1 - alpha_words) * attn_replace
                )
                attn[1:] = attn_replace_new
            else: # SA replacement
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)

            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])

        return attn

    def __init__(
        self,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional["LocalBlendWan"],
        tokenizer,
        device: torch.device,
        low_resource: bool = False,
        max_num_words: int = MAX_NUM_WORDS,
    ):
        super().__init__(low_resource)
        self.batch_size = len(prompts)

        # Compute cross-attention replacement schedule
        self.cross_replace_alpha = ptp_utils_wan.get_time_words_attention_alpha_t5(
            prompts, num_steps, cross_replace_steps, tokenizer, max_num_words
        ).to(device)

        # Self-attention replacement range
        if isinstance(self_replace_steps, float):
            self_replace_steps = (0, self_replace_steps)
        self.num_self_replace = (
            int(num_steps * self_replace_steps[0]),
            int(num_steps * self_replace_steps[1]),
        )

        self.local_blend = local_blend


class AttentionReplaceWan(AttentionControlEditWan):
    """
    Word swap via attention replacement.

    Replaces attention maps to swap words between prompts.
    Example: "a cat sitting" -> "a dog sitting"
    """

    def replace_cross_attention(
        self,
        attn_base: torch.Tensor,
        attn_replace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace cross-attention using word mapper.

        Args:
            attn_base: Base attention [heads, seq_img, seq_text]
            attn_replace: Attention to replace [B-1, heads, seq_img, seq_text]
        """
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(
        self,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        tokenizer,
        device: torch.device,
        local_blend: Optional["LocalBlendWan"] = None,
        low_resource: bool = False,
    ):
        super().__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps,
            local_blend, tokenizer, device, low_resource
        )
        self.mapper = seq_aligner.get_replacement_mapper(
            prompts, tokenizer, max_len=MAX_NUM_WORDS
        ).to(device)


class AttentionRefineWan(AttentionControlEditWan):
    """
    Attention refinement for prompt edits.

    Refines attention when adding/modifying words.
    Example: "a cat" -> "a fluffy cat"
    """

    def replace_cross_attention(
        self,
        attn_base: torch.Tensor,
        attn_replace: torch.Tensor,
    ) -> torch.Tensor:
        """Blend base and replacement attention based on alphas."""
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + attn_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        tokenizer,
        device: torch.device,
        local_blend: Optional["LocalBlendWan"] = None,
        low_resource: bool = False,
    ):
        super().__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps,
            local_blend, tokenizer, device, low_resource
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(
            prompts, tokenizer, max_len=MAX_NUM_WORDS
        )
        self.mapper = self.mapper.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1]).to(device)


class AttentionReweightWan(AttentionControlEditWan):
    """
    Attention reweighting for emphasis control.

    Reweight attention to specific words.
    Example: Emphasize "red" in "a red car"
    """

    def replace_cross_attention(
        self,
        attn_base: torch.Tensor,
        attn_replace: torch.Tensor,
    ) -> torch.Tensor:
        """Apply equalizer weights to attention."""
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, attn_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(
        self,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer: torch.Tensor,
        tokenizer,
        device: torch.device,
        local_blend: Optional["LocalBlendWan"] = None,
        controller: Optional[AttentionControlEditWan] = None,
        low_resource: bool = False,
    ):
        super().__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps,
            local_blend, tokenizer, device, low_resource
        )
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer_wan(
    text: str,
    word_select: Union[int, str, Tuple],
    values: Union[List[float], Tuple[float, ...]],
    tokenizer,
) -> torch.Tensor:
    """
    Create equalizer tensor for attention reweighting.

    Args:
        text: Input prompt
        word_select: Word(s) to reweight
        values: Weight values
        tokenizer: T5 tokenizer

    Returns:
        Equalizer tensor [num_values, MAX_NUM_WORDS]
    """
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)

    equalizer = torch.ones(len(values), MAX_NUM_WORDS)
    values = torch.tensor(values, dtype=torch.float32)

    for word in word_select:
        inds = ptp_utils_wan.get_word_inds_t5(text, word, tokenizer)
        if len(inds) > 0:
            equalizer[:, inds] = values.unsqueeze(1)

    return equalizer


# ============================================================================
# Local Blend for Video
# ============================================================================

class LocalBlendWan:
    """
    Local blending for Wan2.1 video generation.

    Extends blending to the temporal dimension for video editing.
    """

    def __call__(
        self,
        x_t: torch.Tensor,
        attention_store: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Apply local blending to video latents.

        Args:
            x_t: Video latent [B, C, T, H, W]
            attention_store: Stored attention maps

        Returns:
            Blended latent
        """
        k = 1

        # Get cross-attention maps
        maps = attention_store.get("blocks_cross", [])
        if len(maps) == 0:
            return x_t

        # Select middle blocks (most semantic)
        num_maps = len(maps)
        start_idx = num_maps // 3
        end_idx = 2 * num_maps // 3
        maps = maps[start_idx:end_idx]

        if len(maps) == 0:
            return x_t

        # Get dimensions
        batch_size = x_t.shape[0]
        num_frames = x_t.shape[2]
        height = x_t.shape[3]
        width = x_t.shape[4]

        # Process attention maps
        # Each map: [B*heads, num_patches, text_seq]
        processed_maps = []
        for m in maps:
            m = m.to(x_t.device)
            # Reshape: [B*heads, patches, text] -> [B, heads, patches, text]
            num_heads = m.shape[0] // batch_size
            m = m.reshape(batch_size, num_heads, m.shape[1], m.shape[2])
            processed_maps.append(m)

        # Average across maps and heads
        maps_tensor = torch.stack(processed_maps, dim=0)  # [num_maps, B, heads, patches, text]
        maps_tensor = maps_tensor.mean(dim=(0, 2))  # [B, patches, text]

        # Apply word mask
        alpha = self.alpha_layers.to(x_t.device)  # [B, 1, 1, 1, text]
        alpha = alpha.squeeze(1).squeeze(1).squeeze(1)  # [B, text]

        # Weight by alpha and sum
        mask = (maps_tensor * alpha.unsqueeze(1)).sum(-1)  # [B, patches]

        # Infer spatial dimensions (approximate)
        num_patches = mask.shape[1]
        # For Wan with patch_size (1, 2, 2)
        spatial_patches = num_patches // num_frames if num_frames > 0 else num_patches
        h_patches = int(np.sqrt(spatial_patches * height / width))
        w_patches = spatial_patches // h_patches if h_patches > 0 else 1
        t_patches = num_patches // (h_patches * w_patches) if h_patches * w_patches > 0 else 1

        # Reshape mask to 3D
        try:
            mask = mask.reshape(batch_size, t_patches, h_patches, w_patches)
        except:
            # Fallback: use simple interpolation
            mask = mask.reshape(batch_size, 1, 1, -1)

        # Interpolate to latent size
        mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=(num_frames, height, width),
            mode='trilinear',
            align_corners=False,
        ).squeeze(1)

        # Normalize
        mask = mask / mask.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        mask = mask.gt(self.threshold).float()

        # Combine masks and blend
        mask = (mask[:1] + mask[1:]).clamp(max=1)
        x_t = x_t[:1] + mask.unsqueeze(1) * (x_t - x_t[:1])

        return x_t

    def __init__(
        self,
        prompts: List[str],
        words: List[List[str]],
        tokenizer,
        device: torch.device,
        threshold: float = 0.3,
        max_num_words: int = MAX_NUM_WORDS,
    ):
        """
        Initialize LocalBlend.

        Args:
            prompts: List of prompts
            words: Words to blend for each prompt
            tokenizer: T5 tokenizer
            device: Target device
            threshold: Blending threshold
            max_num_words: Maximum text sequence length
        """
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, max_num_words)

        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = ptp_utils_wan.get_word_inds_t5(prompt, word, tokenizer)
                if len(ind) > 0:
                    alpha_layers[i, :, :, :, ind] = 1

        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


# ============================================================================
# Visualization Functions
# ============================================================================

def aggregate_attention_wan(
    attention_store: AttentionStoreWan,
    prompts: List[str],
    is_cross: bool = True,
    select: int = 0,
) -> Optional[torch.Tensor]:
    """
    Aggregate attention maps for visualization.

    Args:
        attention_store: AttentionStoreWan instance
        prompts: List of prompts
        is_cross: Whether to get cross or self attention
        select: Which prompt to visualize

    Returns:
        Aggregated attention tensor or None
    """
    attention_maps = attention_store.get_average_attention()
    key = "blocks_cross" if is_cross else "blocks_self"

    maps = attention_maps.get(key, [])
    if len(maps) == 0:
        return None

    # Stack and average
    out = []
    for item in maps:
        # item: [B*heads, seq_q, seq_kv]
        out.append(item)

    out = torch.stack(out, dim=0)  # [num_layers, B*heads, seq_q, seq_kv]
    out = out.mean(0)  # Average over layers: [B*heads, seq_q, seq_kv]

    # Split batch
    batch_size = len(prompts)
    heads = out.shape[0] // batch_size
    out = out.reshape(batch_size, heads, out.shape[1], out.shape[2])

    # Select prompt and average over heads
    out = out[select].mean(0)  # [seq_q, seq_kv]

    return out


def show_cross_attention_video(
    attention_store: AttentionStoreWan,
    prompts: List[str],
    tokenizer,
    select: int = 0,
    num_tokens: int = 10,
    save_path: Optional[str] = None,
):
    """
    Visualize cross-attention maps for video.

    Args:
        attention_store: AttentionStoreWan instance
        prompts: List of prompts
        tokenizer: T5 tokenizer
        select: Which prompt to visualize
        num_tokens: Number of tokens to display
        save_path: Optional path to save figure
    """
    attn = aggregate_attention_wan(attention_store, prompts, is_cross=True, select=select)

    if attn is None:
        print("No attention maps available")
        return

    # Encode prompt to get tokens
    tokens = tokenizer.encode(prompts[select])[:num_tokens]

    # attn shape: [seq_patches, seq_text]
    num_patches = attn.shape[0]

    # Assume roughly square spatial arrangement
    h = w = int(np.sqrt(num_patches))
    if h * w != num_patches:
        # Try to find factors
        for i in range(int(np.sqrt(num_patches)), 0, -1):
            if num_patches % i == 0:
                h = i
                w = num_patches // i
                break

    fig, axes = plt.subplots(1, len(tokens), figsize=(len(tokens) * 2, 2))

    for i, token_id in enumerate(tokens):
        ax = axes[i] if len(tokens) > 1 else axes

        # Get attention for this token
        token_attn = attn[:, i].reshape(h, w).cpu().numpy()
        token_attn = token_attn / (token_attn.max() + 1e-8)

        ax.imshow(token_attn, cmap='hot')
        ax.set_title(tokenizer.decode([token_id])[:10], fontsize=8)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")

    plt.show()


# ============================================================================
# Main Run Functions
# ============================================================================

def run_and_display_video(
    prompts: List[str],
    controller: AttentionControlWan,
    model,
    latent: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    num_inference_steps: int = NUM_DIFFUSION_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    num_frames: int = 81,
    height: int = 480,
    width: int = 832,
    run_baseline: bool = False,
    low_resource: bool = False,
    offload: bool = False,
):
    """
    Run video generation with P2P and display results.

    Args:
        prompts: List of prompts
        controller: AttentionControlWan instance
        model: WanPipeline instance
        latent: Optional initial latent
        generator: Random generator
        num_inference_steps: Number of diffusion steps
        guidance_scale: CFG scale
        num_frames: Number of video frames
        height: Video height
        width: Video width
        run_baseline: Whether to run without P2P first
        low_resource: Low resource mode
        offload: Offload text_encoder/VAE to CPU during diffusion

    Returns:
        Tuple of (videos, latent)
    """
    if run_baseline:
        print("Running baseline (without P2P)...")
        videos_baseline, latent = run_and_display_video(
            prompts,
            EmptyControlWan(low_resource),
            model,
            latent=latent,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            height=height,
            width=width,
            run_baseline=False,
            low_resource=low_resource,
            offload=offload,
        )
        print("Running with P2P...")

    videos, x_t = ptp_utils_wan.text2video_wan(
        model=model,
        prompt=prompts,
        controller=controller,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        height=height,
        width=width,
        generator=generator,
        latents=latent,
        return_latents=True,
        offload=offload,
    )

    # Display frames
    ptp_utils_wan.view_video_frames(videos)

    return videos, x_t
