#!/usr/bin/env python3
"""
Prompt-to-Prompt examples for Wan2.1-T2V-1.3B video generation.

Demonstrates all P2P operations:
  - Cross-attention visualization
  - Word replacement (swap)
  - Prompt refinement
  - Attention reweighting

Usage:
    python run_examples_wan.py --example all
    python run_examples_wan.py --example cross_attention --num-steps 20 --num-frames 17
    python run_examples_wan.py --example replacement
    python run_examples_wan.py --example refinement
    python run_examples_wan.py --example reweight
    python run_examples_wan.py --example all --offload  # save VRAM by offloading text encoder & VAE
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import ptp_wan
import ptp_utils_wan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_controller(
    kind: str,
    prompts: list,
    tokenizer,
    device: torch.device,
    num_steps: int,
    *,
    cross_replace_steps: float = 0.8,
    self_replace_steps: float = 0.4,
    local_blend=None,
    equalizer=None,
):
    """Convenience factory for the different P2P controllers."""
    if kind == "replace":
        return ptp_wan.AttentionReplaceWan(
            prompts,
            num_steps=num_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            tokenizer=tokenizer,
            device=device,
            local_blend=local_blend,
        )
    elif kind == "refine":
        return ptp_wan.AttentionRefineWan(
            prompts,
            num_steps=num_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            tokenizer=tokenizer,
            device=device,
            local_blend=local_blend,
        )
    elif kind == "reweight":
        assert equalizer is not None, "equalizer required for reweight"
        return ptp_wan.AttentionReweightWan(
            prompts,
            num_steps=num_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            equalizer=equalizer,
            tokenizer=tokenizer,
            device=device,
            local_blend=local_blend,
        )
    else:
        raise ValueError(f"Unknown controller kind: {kind}")


def save_outputs(videos, prompts, tag, output_dir, tokenizer, attention_store=None):
    """Save video mp4s, frame grids, and optionally attention maps."""
    for i, (video, prompt) in enumerate(zip(videos, prompts)):
        prefix = f"{tag}_prompt{i}"

        # Save mp4
        mp4_path = os.path.join(output_dir, f"{prefix}.mp4")
        ptp_utils_wan.export_video(video, mp4_path)
        print(f"  Saved video: {mp4_path}")

        # Save frame grid
        grid_path = os.path.join(output_dir, f"{prefix}_frames.png")
        ptp_utils_wan.view_video_frames(video, save_path=grid_path)
        print(f"  Saved frames: {grid_path}")

    if attention_store is not None:
        attn_path = os.path.join(output_dir, f"{tag}_attention.png")
        ptp_wan.show_cross_attention_video(
            attention_store,
            prompts,
            tokenizer,
            select=0,
            save_path=attn_path,
        )
        print(f"  Saved attention: {attn_path}")


# ---------------------------------------------------------------------------
# Example functions
# ---------------------------------------------------------------------------

def example_cross_attention(model, tokenizer, device, args):
    """Generate a single video and visualize cross-attention maps."""
    print("\n=== Cross-Attention Visualization ===")
    prompts = ["A cat walking in the garden"]

    g = torch.Generator(device=device).manual_seed(args.seed)
    controller = ptp_wan.AttentionStoreWan(low_resource=False)

    videos, latent = ptp_wan.run_and_display_video(
        prompts,
        controller,
        model,
        generator=g,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        offload=args.offload,
    )

    save_outputs(
        [videos[0]] if videos.ndim == 5 else [videos],
        prompts,
        "cross_attention",
        args.output_dir,
        tokenizer,
        attention_store=controller,
    )
    print("Cross-attention example done.\n")


def example_replacement(model, tokenizer, device, args):
    """Word swap: cat->dog, garden->beach (with LocalBlend)."""
    print("\n=== Word Replacement (Swap) ===")

    g = torch.Generator(device=device).manual_seed(args.seed)

    # --- Simple swap: cat -> dog ---
    prompts_animal = [
        "A cat walking in the garden",
        "A dog walking in the garden",
    ]
    controller = make_controller(
        "replace", prompts_animal, tokenizer, device, args.num_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
    )
    videos, latent = ptp_wan.run_and_display_video(
        prompts_animal,
        controller,
        model,
        generator=g,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        offload=args.offload,
    )
    save_outputs(videos, prompts_animal, "replace_cat_dog", args.output_dir, tokenizer)

    # --- Swap with LocalBlend: garden -> beach ---
    prompts_scene = [
        "A cat walking in the garden",
        "A cat walking on the beach",
    ]
    lb = ptp_wan.LocalBlendWan(
        prompts_scene,
        words=[["garden"], ["beach"]],
        tokenizer=tokenizer,
        device=device,
    )
    g = torch.Generator(device=device).manual_seed(args.seed)
    controller = make_controller(
        "replace", prompts_scene, tokenizer, device, args.num_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        local_blend=lb,
    )
    videos, latent = ptp_wan.run_and_display_video(
        prompts_scene,
        controller,
        model,
        generator=g,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        offload=args.offload,
    )
    save_outputs(videos, prompts_scene, "replace_garden_beach", args.output_dir, tokenizer)
    print("Replacement example done.\n")


def example_refinement(model, tokenizer, device, args):
    """Prompt refinement: add / change words while preserving layout."""
    print("\n=== Prompt Refinement ===")

    g = torch.Generator(device=device).manual_seed(args.seed)

    # --- Add adjective: "A cat" -> "A fluffy cat" ---
    prompts_adj = [
        "A cat sitting on a couch",
        "A fluffy cat sitting on a couch",
    ]
    controller = make_controller(
        "refine", prompts_adj, tokenizer, device, args.num_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
    )
    videos, latent = ptp_wan.run_and_display_video(
        prompts_adj,
        controller,
        model,
        generator=g,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        offload=args.offload,
    )
    save_outputs(videos, prompts_adj, "refine_fluffy", args.output_dir, tokenizer)

    # --- Style change: add "in winter" ---
    prompts_season = [
        "A house on a hill",
        "A house on a hill in winter",
    ]
    g = torch.Generator(device=device).manual_seed(args.seed)
    controller = make_controller(
        "refine", prompts_season, tokenizer, device, args.num_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
    )
    videos, latent = ptp_wan.run_and_display_video(
        prompts_season,
        controller,
        model,
        generator=g,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        offload=args.offload,
    )
    save_outputs(videos, prompts_season, "refine_winter", args.output_dir, tokenizer)
    print("Refinement example done.\n")


def example_reweight(model, tokenizer, device, args):
    """Attention reweighting: emphasize / de-emphasize words."""
    print("\n=== Attention Reweighting ===")

    g = torch.Generator(device=device).manual_seed(args.seed)

    prompt = "A red car driving on the road"
    prompts = [prompt, prompt]

    # Emphasize "red" (scale x5)
    equalizer = ptp_wan.get_equalizer_wan(prompt, "red", (5.0,), tokenizer)
    controller = make_controller(
        "reweight", prompts, tokenizer, device, args.num_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        equalizer=equalizer,
    )
    videos, latent = ptp_wan.run_and_display_video(
        prompts,
        controller,
        model,
        generator=g,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        offload=args.offload,
    )
    save_outputs(videos, prompts, "reweight_red_up", args.output_dir, tokenizer)

    # De-emphasize "red" (scale x-5)
    g = torch.Generator(device=device).manual_seed(args.seed)
    equalizer = ptp_wan.get_equalizer_wan(prompt, "red", (-5.0,), tokenizer)
    controller = make_controller(
        "reweight", prompts, tokenizer, device, args.num_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        equalizer=equalizer,
    )
    videos, latent = ptp_wan.run_and_display_video(
        prompts,
        controller,
        model,
        generator=g,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        offload=args.offload,
    )
    save_outputs(videos, prompts, "reweight_red_down", args.output_dir, tokenizer)
    print("Reweighting example done.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXAMPLES = {
    "cross_attention": example_cross_attention,
    "replacement": example_replacement,
    "refinement": example_refinement,
    "reweight": example_reweight,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt-to-Prompt examples for Wan2.1-T2V-1.3B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--example",
        type=str,
        default="all",
        choices=list(EXAMPLES.keys()) + ["all"],
        help="Which example to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ptp_outputs_wan",
        help="Directory for output files (default: ./ptp_outputs_wan)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of diffusion steps (default: 20)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=17,
        help="Number of video frames to generate (default: 17)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Offload text encoder and VAE to CPU when not in use (saves VRAM)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    ptp_utils_wan.set_save_dir(args.output_dir)

    print(f"Output directory: {args.output_dir}")
    print(f"Steps: {args.num_steps}  Frames: {args.num_frames}  "
          f"Resolution: {args.height}x{args.width}  Seed: {args.seed}  "
          f"Offload: {args.offload}")

    # Load model
    print("\nLoading Wan2.1-T2V-1.3B model...")
    model, tokenizer, device = ptp_wan.load_wan_model(enable_offload=args.offload)
    print(f"Model loaded on {device}\n")

    # Run selected examples
    if args.example == "all":
        for name, fn in EXAMPLES.items():
            print(f"--- Running: {name} ---")
            fn(model, tokenizer, device, args)
    else:
        EXAMPLES[args.example](model, tokenizer, device, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
