# Copyright 2022 Google LLC
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
Prompt-to-Prompt Examples with Stable Diffusion

Converted from prompt-to-prompt_stable.ipynb

Usage:
    python run_examples.py --example cross_attention
    python run_examples.py --example replacement
    python run_examples.py --example refinement
    python run_examples.py --example reweight
    python run_examples.py --example all
    
    # Save to specific directory without displaying
    python run_examples.py --example all --output-dir ./my_outputs --no-show
"""

import argparse
import torch
import matplotlib
import os

from ptp_stable import (
    load_model,
    LocalBlend,
    AttentionStore,
    AttentionReplace,
    AttentionRefine,
    AttentionReweight,
    EmptyControl,
    get_equalizer,
    run_and_display,
    show_cross_attention,
    NUM_DIFFUSION_STEPS,
    LOW_RESOURCE,
)
import ptp_utils


def example_cross_attention(ldm_stable, tokenizer, device, x_t=None):
    """Cross-Attention Visualization Example"""
    print("\n" + "="*60)
    print("Cross-Attention Visualization")
    print("="*60)
    
    g_cpu = torch.Generator().manual_seed(8888)
    prompts = ["A painting of a squirrel eating a burger"]
    controller = AttentionStore(low_resource=LOW_RESOURCE)
    image, x_t = run_and_display(prompts, controller, ldm_stable, latent=None, run_baseline=False, generator=g_cpu)
    show_cross_attention(prompts, tokenizer, controller, res=32, from_where=("up", "down"))
    
    return x_t


def example_replacement(ldm_stable, tokenizer, device, x_t):
    """Replacement Edit Examples"""
    print("\n" + "="*60)
    print("Replacement Edit")
    print("="*60)
    
    # Basic replacement
    print("\n--- Basic Replacement: squirrel -> lion ---")
    prompts = ["A painting of a squirrel eating a burger",
               "A painting of a lion eating a burger"]
    controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4,
                                  tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=True)
    
    # Modify cross-attention injection steps for specific words
    print("\n--- Reduced restriction on 'lion' ---")
    prompts = ["A painting of a squirrel eating a burger",
               "A painting of a lion eating a burger"]
    controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps={"default_": 1., "lion": .4},
                                  self_replace_steps=0.4, tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)
    
    # Local edit
    print("\n--- Local Edit: preserve burger ---")
    prompts = ["A painting of a squirrel eating a burger",
               "A painting of a lion eating a burger"]
    lb = LocalBlend(prompts, ("squirrel", "lion"), tokenizer, device)
    controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS,
                                  cross_replace_steps={"default_": 1., "lion": .4},
                                  self_replace_steps=0.4, local_blend=lb,
                                  tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)
    
    # Another local edit example
    print("\n--- Local Edit: burger -> lasagne ---")
    prompts = ["A painting of a squirrel eating a burger",
               "A painting of a squirrel eating a lasagne"]
    lb = LocalBlend(prompts, ("burger", "lasagne"), tokenizer, device)
    controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS,
                                  cross_replace_steps={"default_": 1., "lasagne": .2},
                                  self_replace_steps=0.4,
                                  local_blend=lb,
                                  tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=True)


def example_refinement(ldm_stable, tokenizer, device, x_t):
    """Refinement Edit Examples"""
    print("\n" + "="*60)
    print("Refinement Edit")
    print("="*60)
    
    # Neoclassical style
    print("\n--- Style refinement: neoclassical ---")
    prompts = ["A painting of a squirrel eating a burger",
               "A neoclassical painting of a squirrel eating a burger"]
    controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS,
                                 cross_replace_steps=.5, 
                                 self_replace_steps=.2,
                                 tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t)
    
    # Season change - fall
    print("\n--- Season change: fall ---")
    prompts = ["a photo of a house on a mountain",
               "a photo of a house on a mountain at fall"]
    controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                 self_replace_steps=.4,
                                 tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t)
    
    # Season change - winter
    print("\n--- Season change: winter ---")
    prompts = ["a photo of a house on a mountain",
               "a photo of a house on a mountain at winter"]
    controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                 self_replace_steps=.4,
                                 tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t)
    
    # Pea soup
    print("\n--- Refinement: soup -> pea soup ---")
    prompts = ["soup",
               "pea soup"] 
    lb = LocalBlend(prompts, ("soup", "soup"), tokenizer, device)
    controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                 self_replace_steps=.4,
                                 local_blend=lb,
                                 tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)


def example_reweight(ldm_stable, tokenizer, device, x_t):
    """Attention Re-Weighting Examples"""
    print("\n" + "="*60)
    print("Attention Re-Weighting")
    print("="*60)
    
    # Smiling bunny - increase attention
    print("\n--- Increase attention to 'smiling' ---")
    prompts = ["a smiling bunny doll"] * 2
    # pay 5 times more attention to the word "smiling"
    equalizer = get_equalizer(prompts[1], ("smiling",), (5,), tokenizer)
    controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                   self_replace_steps=.4,
                                   equalizer=equalizer,
                                   tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)
    
    # Pink bear - reduce pink on bicycle
    print("\n--- Reduce 'pink' on bicycle (local edit) ---")
    prompts = ["pink bear riding a bicycle"] * 2
    # pay less attention to the word "pink"
    equalizer = get_equalizer(prompts[1], ("pink",), (-1,), tokenizer)
    # apply the edit on the bikes 
    lb = LocalBlend(prompts, ("bicycle", "bicycle"), tokenizer, device)
    controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                   self_replace_steps=.4,
                                   equalizer=equalizer,
                                   local_blend=lb,
                                   tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)
    
    # Pea soup with croutons
    print("\n--- Refinement: soup -> pea soup with croutons ---")
    prompts = ["soup",
               "pea soup with croutons"] 
    lb = LocalBlend(prompts, ("soup", "soup"), tokenizer, device)
    controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                 self_replace_steps=.4, local_blend=lb,
                                 tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)
    
    # With more attention to croutons
    print("\n--- With more attention to 'croutons' ---")
    prompts = ["soup",
               "pea soup with croutons"] 
    lb = LocalBlend(prompts, ("soup", "soup"), tokenizer, device)
    controller_a = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, 
                                   self_replace_steps=.4, local_blend=lb,
                                   tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    # pay 3 times more attention to the word "croutons"
    equalizer = get_equalizer(prompts[1], ("croutons",), (3,), tokenizer)
    controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                   self_replace_steps=.4, equalizer=equalizer, local_blend=lb,
                                   controller=controller_a,
                                   tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)
    
    # Fried potatoes
    print("\n--- Refinement: potatos -> fried potatos ---")
    prompts = ["potatos",
               "fried potatos"] 
    lb = LocalBlend(prompts, ("potatos", "potatos"), tokenizer, device)
    controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                 self_replace_steps=.4, local_blend=lb,
                                 tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)
    
    # With more attention to fried
    print("\n--- With more attention to 'fried' ---")
    prompts = ["potatos",
               "fried potatos"] 
    lb = LocalBlend(prompts, ("potatos", "potatos"), tokenizer, device)
    controller_a = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, 
                                   self_replace_steps=.4, local_blend=lb,
                                   tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    # pay 10 times more attention to the word "fried"
    equalizer = get_equalizer(prompts[1], ("fried",), (10,), tokenizer)
    controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                   self_replace_steps=.4, equalizer=equalizer, local_blend=lb,
                                   controller=controller_a,
                                   tokenizer=tokenizer, device=device, low_resource=LOW_RESOURCE)
    _ = run_and_display(prompts, controller, ldm_stable, latent=x_t, run_baseline=False)


def main():
    parser = argparse.ArgumentParser(description='Prompt-to-Prompt Examples with Stable Diffusion')
    parser.add_argument('--example', type=str, default='all',
                        choices=['cross_attention', 'replacement', 'refinement', 'reweight', 'all'],
                        help='Which example to run')
    parser.add_argument('--model', type=str, default='CompVis/stable-diffusion-v1-4',
                        help='Model ID to use')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace auth token')
    parser.add_argument('--low-resource', action='store_true',
                        help='Enable low resource mode for 12GB GPU')
    parser.add_argument('--output-dir', type=str, default='./ptp_outputs',
                        help='Directory to save output images')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display images (useful for headless servers)')
    args = parser.parse_args()
    
    # Set up matplotlib backend for headless mode
    if args.no_show:
        matplotlib.use('Agg')
    
    # Set output directory
    ptp_utils.set_save_dir(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output images will be saved to: {args.output_dir}")
    
    # Monkey-patch view_images to respect --no-show
    original_view_images = ptp_utils.view_images
    def patched_view_images(*args_inner, **kwargs_inner):
        kwargs_inner['show'] = not args.no_show
        return original_view_images(*args_inner, **kwargs_inner)
    ptp_utils.view_images = patched_view_images
    
    # Load model
    print("Loading Stable Diffusion model...")
    ldm_stable, tokenizer, device = load_model(model_id=args.model, auth_token=args.token)
    print(f"Model loaded on device: {device}")
    
    # Run cross attention first to get initial latent
    x_t = None
    
    if args.example in ['cross_attention', 'all']:
        x_t = example_cross_attention(ldm_stable, tokenizer, device, x_t)
    
    # If we don't have x_t yet, generate it
    if x_t is None:
        print("\nGenerating initial latent...")
        g_cpu = torch.Generator().manual_seed(8888)
        prompts = ["A painting of a squirrel eating a burger"]
        controller = AttentionStore(low_resource=LOW_RESOURCE)
        _, x_t = run_and_display(prompts, controller, ldm_stable, latent=None, run_baseline=False, generator=g_cpu)
    
    if args.example in ['replacement', 'all']:
        example_replacement(ldm_stable, tokenizer, device, x_t)
    
    if args.example in ['refinement', 'all']:
        example_refinement(ldm_stable, tokenizer, device, x_t)
    
    if args.example in ['reweight', 'all']:
        example_reweight(ldm_stable, tokenizer, device, x_t)
    
    print("\n" + "="*60)
    print("Examples completed!")
    print(f"Output images saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
