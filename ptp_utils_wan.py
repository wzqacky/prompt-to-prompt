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
Utility functions for Prompt-to-Prompt with Wan2.1 Video Generation Model.

This module provides the core infrastructure for attention manipulation
in the Wan2.1 DiT-based video generation model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Callable, Union
from tqdm import tqdm
import os

# Global save directory
_save_dir = "./ptp_outputs_wan"
_video_counter = 0


def set_save_dir(save_dir: str):
    """Set the directory for saving output videos."""
    global _save_dir
    _save_dir = save_dir
    os.makedirs(_save_dir, exist_ok=True)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


# ============================================================================
# P2P Attention Processor for Wan2.1
# ============================================================================

class P2PWanAttnProcessor:
    """
    Custom attention processor for Wan2.1 that hooks into P2P controller.

    This processor wraps the attention computation and allows the P2P
    controller to manipulate attention probabilities.
    """

    def __init__(self, controller, block_idx: int, attn_type: str):
        """
        Initialize P2P attention processor.

        Args:
            controller: P2P controller instance
            block_idx: Index of the transformer block
            attn_type: "self" or "cross"
        """
        self.controller = controller
        self.block_idx = block_idx
        self.attn_type = attn_type
        self._attention_backend = None
        self._parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with P2P attention manipulation.
        """
        is_cross = encoder_hidden_states is not None

        # Handle I2V case (image conditioning)
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Get Q, K, V projections
        if encoder_hidden_states is None:
            encoder_hidden_states_input = hidden_states
        else: # CA
            encoder_hidden_states_input = encoder_hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states_input)
        value = attn.to_v(encoder_hidden_states_input)

        # Apply normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape: [B, seq, dim] -> [B, seq, heads, head_dim]
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply rotary embeddings for self-attention
        if rotary_emb is not None:
            def apply_rotary_emb_local(hidden_states, freqs_cos, freqs_sin):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb_local(query, *rotary_emb)
            key = apply_rotary_emb_local(key, *rotary_emb)

        # Compute attention scores manually for P2P hook
        # [B, seq_q, heads, head_dim] @ [B, seq_k, heads, head_dim].T
        # -> [B, heads, seq_q, seq_k]
        batch_size, seq_q, heads, head_dim = query.shape
        seq_k = key.shape[1]

        # Transpose: [B, seq, heads, dim] -> [B, heads, seq, dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute attention scores
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale

        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = attn_scores.softmax(dim=-1).to(value.dtype)

        # === P2P HOOK: Manipulate attention here ===
        # Reshape for controller: [B, heads, seq_q, seq_k] -> [B*heads, seq_q, seq_k]
        attn_probs_reshaped = attn_probs.reshape(batch_size * heads, seq_q, seq_k)
        attn_probs_reshaped = self.controller(attn_probs_reshaped, is_cross, self.block_idx)
        attn_probs = attn_probs_reshaped.reshape(batch_size, heads, seq_q, seq_k)

        # Compute output
        hidden_states = torch.matmul(attn_probs, value)

        # Reshape: [B, heads, seq, dim] -> [B, seq, heads*dim]
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # Handle I2V case
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            attn_scores_img = torch.matmul(query, key_img.transpose(-1, -2)) * scale
            attn_probs_img = attn_scores_img.softmax(dim=-1).to(value_img.dtype)
            hidden_states_img = torch.matmul(attn_probs_img, value_img)
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states + hidden_states_img.type_as(hidden_states)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control_wan(model, controller, cross_only: bool = True):
    """
    Register P2P attention processors for Wan2.1 DiT transformer.

    Args:
        model: WanPipeline instance
        controller: AttentionControl instance
        cross_only: When True (default), only hook cross-attention layers to save memory

    Returns:
        int: Number of attention layers registered
    """
    class DummyController:
        def __call__(self, attn, is_cross, block_idx):
            return attn
        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    transformer = model.transformer

    if hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    elif hasattr(transformer, 'transformer_blocks'):
        blocks = transformer.transformer_blocks
    else:
        raise ValueError("Cannot find transformer blocks in model")

    # Store original processors for cleanup
    if not hasattr(model, '_ptp_original_processors'):
        model._ptp_original_processors = {}

    att_count = 0

    for idx, block in enumerate(blocks):
        # Self-attention (attn1) — skip by default to keep flash attention
        if not cross_only and hasattr(block, 'attn1') and block.attn1 is not None:
            model._ptp_original_processors[(idx, 'attn1')] = block.attn1.processor
            block.attn1.processor = P2PWanAttnProcessor(controller, idx, "self")
            att_count += 1

        # Cross-attention (attn2)
        if hasattr(block, 'attn2') and block.attn2 is not None:
            model._ptp_original_processors[(idx, 'attn2')] = block.attn2.processor
            block.attn2.processor = P2PWanAttnProcessor(controller, idx, "cross")
            att_count += 1

    controller.num_att_layers = att_count
    print(f"Registered {att_count} attention layers for P2P control"
          f" (cross_only={cross_only})")

    return att_count


def unregister_attention_control_wan(model):
    """Restore original attention processors."""
    if not hasattr(model, '_ptp_original_processors'):
        return

    transformer = model.transformer

    if hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    elif hasattr(transformer, 'transformer_blocks'):
        blocks = transformer.transformer_blocks
    else:
        return

    for (idx, attn_name), processor in model._ptp_original_processors.items():
        block = blocks[idx]
        attn = getattr(block, attn_name)
        if attn is not None:
            attn.processor = processor

    delattr(model, '_ptp_original_processors')


# ============================================================================
# Diffusion Step for Wan2.1
# ============================================================================

def diffusion_step_wan(
    model,
    controller,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    guidance_scale: float = 5.0,
    prompt_embeds_null: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single diffusion step for Wan2.1 with P2P control."""
    if prompt_embeds_null is None:
        prompt_embeds_null = torch.zeros_like(prompt_embeds)

    latent_model_input = torch.cat([latents] * 2)
    prompt_embeds_input = torch.cat([prompt_embeds_null, prompt_embeds])

    timestep_input = timestep
    if not isinstance(timestep, torch.Tensor):
        timestep_input = torch.tensor([timestep], device=latents.device, dtype=latents.dtype)
    if timestep_input.dim() == 0:
        timestep_input = timestep_input.unsqueeze(0)
    timestep_input = timestep_input.expand(latent_model_input.shape[0])

    with torch.autocast("cuda", dtype=latents.dtype):
        noise_pred = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds_input,
            return_dict=False,
        )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = model.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    latents = controller.step_callback(latents)

    return latents


# ============================================================================
# Video Generation with P2P
# ============================================================================

@torch.no_grad()
def text2video_wan(
    model,
    prompt: Union[str, List[str]],
    controller,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_frames: int = 81,
    height: int = 480,
    width: int = 832,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    return_latents: bool = False,
    max_sequence_length: int = 512,
    offload: bool = False,
) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
    """
    Generate video with P2P attention control.

    Args:
        offload: When True, offload text_encoder and VAE to CPU when not in
                 use so only the transformer occupies GPU memory during the
                 diffusion loop.  After decoding the transformer is moved
                 back to GPU so subsequent calls work as expected.
    """
    # Derive device from transformer (reliable even when other components are offloaded)
    device = next(model.transformer.parameters()).device
    dtype = model.transformer.dtype

    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)

    if negative_prompt is None:
        negative_prompt = [""] * batch_size
    elif isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size

    register_attention_control_wan(model, controller)

    # ------------------------------------------------------------------
    # Text encoding  (text_encoder needed on GPU)
    # ------------------------------------------------------------------
    if offload:
        model.text_encoder.to(device)

    text_inputs = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_inputs_ids, mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = model.text_encoder(text_inputs_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.shape[0], u.shape[1])]) for u in prompt_embeds], dim=0
    )

    negative_text_inputs = model.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    neg_ids, neg_mask = negative_text_inputs.input_ids.to(device), negative_text_inputs.attention_mask.to(device)
    neg_seq_lens = neg_mask.gt(0).sum(dim=1).long()

    negative_prompt_embeds = model.text_encoder(neg_ids, neg_mask).last_hidden_state
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)
    negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, neg_seq_lens)]
    negative_prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.shape[0], u.shape[1])]) for u in negative_prompt_embeds], dim=0
    )

    # ------------------------------------------------------------------
    # Offload text encoder – no longer needed until next call
    # ------------------------------------------------------------------
    if offload:
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Prepare latents
    # ------------------------------------------------------------------
    vae_temporal_compression = getattr(model.vae.config, 'temporal_compression_ratio', 4)
    vae_spatial_compression = getattr(model.vae.config, 'spatial_compression_ratio', 8)

    latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
    latent_height = height // vae_spatial_compression
    latent_width = width // vae_spatial_compression
    latent_channels = model.transformer.config.in_channels

    if latents is None:
        latents = torch.randn(
            (batch_size, latent_channels, latent_num_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        latents = latents.to(device=device, dtype=dtype)

    latents = latents * getattr(model.scheduler, "init_noise_sigma", 1.0)

    # ------------------------------------------------------------------
    # Diffusion loop  (only transformer on GPU)
    # ------------------------------------------------------------------
    model.scheduler.set_timesteps(num_inference_steps, device=device)

    for i, t in enumerate(tqdm(model.scheduler.timesteps, desc="Video generation")):
        latents = diffusion_step_wan(
            model=model,
            controller=controller,
            latents=latents,
            prompt_embeds=prompt_embeds,
            timestep=t,
            guidance_scale=guidance_scale,
            prompt_embeds_null=negative_prompt_embeds,
        )

    # ------------------------------------------------------------------
    # VAE decode  (swap transformer ↔ VAE on GPU)
    # ------------------------------------------------------------------
    if offload:
        model.transformer.to("cpu")
        torch.cuda.empty_cache()
        model.vae.to(device)

    video = latent2video(model.vae, latents)
    unregister_attention_control_wan(model)

    if offload:
        model.vae.to("cpu")
        torch.cuda.empty_cache()
        model.transformer.to(device)

    if return_latents:
        return video, latents
    return video, None


def latent2video(vae, latents: torch.Tensor) -> np.ndarray:
    """Decode video latents using proper Wan2.1 per-channel normalization."""
    latents = latents.to(vae.dtype)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean

    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]

    video = (video / 2 + 0.5).clamp(0, 1)
    # Cast back to f32, as numpy does not support bf16
    video = video.permute(0, 2, 3, 4, 1).cpu().float().numpy()
    video = (video * 255).astype(np.uint8)

    return video

def export_video(frames: np.ndarray, path: str, fps: int = 15):
    """Export video frames to file."""
    try:
        from diffusers.utils import export_to_video
        if frames.ndim == 5:
            frames = frames[0]
        export_to_video(frames, path, fps=fps)
        print(f"Video saved to: {path}")
    except ImportError:
        import imageio
        if frames.ndim == 5:
            frames = frames[0]
        imageio.mimwrite(path, frames, fps=fps)
        print(f"Video saved to: {path}")


def view_video_frames(frames: np.ndarray, num_display: int = 8, save_path: Optional[str] = None):
    """Display selected frames from video."""
    import matplotlib.pyplot as plt

    if frames.ndim == 5:
        frames = frames[0]

    num_frames = frames.shape[0]
    indices = np.linspace(0, num_frames - 1, num_display, dtype=int)

    fig, axes = plt.subplots(1, num_display, figsize=(num_display * 2, 2))
    for i, idx in enumerate(indices):
        axes[i].imshow(frames[idx])
        axes[i].set_title(f"F{idx}")
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Frames saved to: {save_path}")

    plt.show()


# ============================================================================
# Word Index Utilities for T5
# ============================================================================

def get_word_inds_t5(text: str, word_place: Union[int, str], tokenizer) -> np.ndarray:
    """Get token indices for a word in T5-encoded text."""
    split_text = text.split(" ")

    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place.lower() in word.lower()]
    elif isinstance(word_place, int):
        word_place = [word_place]

    out = []
    if len(word_place) > 0:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        tokens_decoded = [tokenizer.decode([t]) for t in encoded]

        current_word_idx = 0
        current_word = ""

        for token_idx, token_text in enumerate(tokens_decoded):
            current_word += token_text

            if current_word_idx < len(split_text):
                target_word = split_text[current_word_idx]

                if current_word_idx in word_place:
                    out.append(token_idx)

                if current_word.strip().endswith(target_word) or len(current_word.strip()) >= len(target_word):
                    current_word_idx += 1
                    current_word = ""

    return np.array(out)


def get_time_words_attention_alpha_t5(
    prompts: List[str],
    num_steps: int,
    cross_replace_steps: Union[float, dict],
    tokenizer,
    max_num_words: int = 512,
) -> torch.Tensor:
    """Compute attention alpha values for time-based word replacement."""
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)

    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)

    for i in range(len(prompts) - 1):
        alpha_time_words = _update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i
        )

    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds_t5(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = _update_alpha_time_word(alpha_time_words, item, i, ind)

    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


def _update_alpha_time_word(alpha, bounds, prompt_ind, word_inds=None):
    """Update alpha values for specific time bounds and word indices."""
    if isinstance(bounds, float):
        bounds = (0, bounds)

    start = int(bounds[0] * alpha.shape[0])
    end = int(bounds[1] * alpha.shape[0])

    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])

    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0

    return alpha
