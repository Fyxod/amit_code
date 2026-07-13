#!/usr/bin/env python
"""
FLUX White-Box Backend Script.

This script provides a CLI interface to the FLUX white-box backend
from the GLASS project. It allows running image edits using the FLUX.2
Klein model with configurable parameters.

Usage:
    python flux.py -i input.jpg -o output.jpg -p "Make him smile"
    python flux.py -i input.jpg -o output.jpg -p "Add glasses" --steps 10 --guidance 2.0
"""

import argparse
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any


@dataclass
class FluxSettings:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    torch_dtype: str = "bfloat16"
    diffusion_steps: int = 4
    guidance_scale: float = 1.0
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, ...] = (9, 18, 27)
    objective_timestep_index: int = 0
    seed: int = 1234


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return aliases[name.lower()]


class FluxBackend:
    name = "flux"

    def __init__(self, device: torch.device, settings: FluxSettings | None = None) -> None:
        self.device = device
        self.settings = settings or FluxSettings()
        self.pipe = self._load()

    def _load(self):
        if self.device.type != "cuda":
            raise RuntimeError("FLUX white-box GLASS runs require CUDA.")
        try:
            from diffusers import Flux2KleinPipeline
        except Exception as error:
            raise RuntimeError(
                "Could not import diffusers.Flux2KleinPipeline. This backend follows the clean/whitebox "
                "FLUX.2 Klein implementation; install the same diffusers version used there."
            ) from error
        pipe = Flux2KleinPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=_dtype(self.settings.torch_dtype),
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for module_name in ("vae", "text_encoder", "transformer"):
            module = getattr(pipe, module_name, None)
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        return pipe

    def _timesteps(self, image_seq_len: int) -> torch.Tensor:
        try:
            from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

            sigmas = np.linspace(1.0, 1.0 / self.settings.diffusion_steps, self.settings.diffusion_steps)
            sigmas = None if getattr(self.pipe.scheduler.config, "use_flow_sigmas", False) else sigmas
            timesteps, _ = retrieve_timesteps(
                self.pipe.scheduler,
                self.settings.diffusion_steps,
                self.device,
                sigmas=sigmas,
                mu=compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.settings.diffusion_steps),
            )
            self.pipe.scheduler.set_begin_index(0)
            return timesteps
        except Exception:
            self.pipe.scheduler.set_timesteps(self.settings.diffusion_steps, device=self.device)
            return self.pipe.scheduler.timesteps

    def _condition(self, image_tensor: torch.Tensor, generator: torch.Generator):
        image = (image_tensor * 2.0 - 1.0).to(self.pipe.vae.dtype)
        return self.pipe.prepare_image_latents(
            images=[image],
            batch_size=1,
            generator=generator,
            device=self.device,
            dtype=self.pipe.vae.dtype,
        )

    def _prediction(self, image_tensor: torch.Tensor, reference: dict[str, Any], generator: torch.Generator):
        conditioning, image_ids = self._condition(image_tensor, generator)
        hidden = torch.cat([reference["latents"], conditioning], dim=1).to(self.pipe.transformer.dtype)
        img_ids = torch.cat([reference["latent_ids"], image_ids], dim=1)
        output = self.pipe.transformer(
            hidden_states=hidden,
            timestep=reference["timestep"],
            guidance=None,
            encoder_hidden_states=reference["prompt_embeds"],
            txt_ids=reference["text_ids"],
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        return output[:, : reference["latents"].shape[1]].float(), conditioning.float()

    def prepare_reference(self, original_tensor: torch.Tensor, prompt: str, objective: str) -> dict[str, Any]:
        generator = torch.Generator(device=self.device).manual_seed(self.settings.seed)
        with torch.no_grad():
            prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                max_sequence_length=self.settings.max_sequence_length,
                text_encoder_out_layers=self.settings.text_encoder_out_layers,
            )
            channels = self.pipe.transformer.config.in_channels // 4
            latents, latent_ids = self.pipe.prepare_latents(
                batch_size=1,
                num_latents_channels=channels,
                height=original_tensor.shape[-2],
                width=original_tensor.shape[-1],
                dtype=prompt_embeds.dtype,
                device=self.device,
                generator=generator,
                latents=None,
            )
            timesteps = self._timesteps(latents.shape[1])
            timestep = timesteps[min(max(0, self.settings.objective_timestep_index), len(timesteps) - 1)]
            timestep = timestep.expand(1).to(prompt_embeds.dtype) / 1000.0
            reference = {
                "prompt": prompt,
                "prompt_embeds": prompt_embeds,
                "text_ids": text_ids,
                "latents": latents.detach(),
                "latent_ids": latent_ids,
                "timestep": timestep,
                "objective": objective,
            }
            clean_pred, clean_cond = self._prediction(original_tensor, reference, generator)
        reference["clean_prediction"] = clean_pred.detach()
        reference["clean_conditioning"] = clean_cond.detach()
        return reference

    def internal_objective(self, perturbed: torch.Tensor, reference: dict[str, Any], objective: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch.nn.functional as F
        generator = torch.Generator(device=self.device).manual_seed(self.settings.seed)
        pred, conditioning = self._prediction(perturbed, reference, generator)
        pred_mse = F.mse_loss(pred.float(), reference["clean_prediction"].float())
        vae_mse = F.mse_loss(conditioning.float(), reference["clean_conditioning"].float())
        if objective == "vae":
            value = vae_mse
        elif objective == "transformer_pred":
            value = pred_mse
        else:
            raise ValueError(f"Unsupported FLUX objective: {objective}")
        return value, {
            "vae_mse": vae_mse,
            "transformer_pred_mse": pred_mse,
        }

    @torch.inference_mode()
    def generate_edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            image=image,
            prompt=prompt,
            height=image.height,
            width=image.width,
            num_inference_steps=self.settings.diffusion_steps,
            guidance_scale=self.settings.guidance_scale,
            generator=generator,
            max_sequence_length=self.settings.max_sequence_length,
            text_encoder_out_layers=self.settings.text_encoder_out_layers,
        )
        return result.images[0].convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FLUX white-box backend for image editing"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path to save output image")
    parser.add_argument("--prompt", "-p", type=str, default="Make him smile",
                        help="Text prompt for the edit")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.2-klein-4B",
                        help="HuggingFace model ID")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Torch dtype for model")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="Guidance scale")
    parser.add_argument("--max-sequence-length", type=int, default=512,
                        help="Max sequence length for text encoding")
    parser.add_argument("--text-encoder-out-layers", type=int, nargs=3, default=[9, 18, 27],
                        help="Text encoder output layers")
    parser.add_argument("--objective-timestep-index", type=int, default=0,
                        help="Timestep index for objective computation")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run on")
    parser.add_argument("--objective", type=str, default=None,
                        choices=["vae", "transformer_pred"],
                        help="Compute internal objective (white-box mode)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load image
    print(f"Loading image: {args.input}")
    image = Image.open(args.input).convert("RGB")
    print(f"  Image size: {image.size}")

    # Create settings and backend
    settings = FluxSettings(
        model_id=args.model_id,
        torch_dtype=args.dtype,
        diffusion_steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        text_encoder_out_layers=tuple(args.text_encoder_out_layers),
        objective_timestep_index=args.objective_timestep_index,
        seed=args.seed,
    )

    print(f"\nInitialising FLUX backend...")
    print(f"  Model: {settings.model_id}")
    print(f"  Steps: {settings.diffusion_steps}")
    print(f"  Guidance: {settings.guidance_scale}")
    print(f"  Dtype: {settings.torch_dtype}")
    print(f"  Max seq len: {settings.max_sequence_length}")

    backend = FluxBackend(device, settings)

    # Run edit or compute objective
    if args.objective is None:
        # Standard image editing
        print(f"\nRunning image edit with prompt: '{args.prompt}'")
        result = backend.generate_edit(image, args.prompt, args.seed)
        result.save(args.output)
        print(f"\nSaved edited image → {args.output}")
    else:
        # White-box objective computation
        print(f"\nComputing white-box objective: {args.objective}")
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        reference = backend.prepare_reference(image_tensor, args.prompt, args.objective)
        value, metrics = backend.internal_objective(image_tensor, reference, args.objective)
        
        print(f"\nObjective value: {value.item():.6f}")
        print("Metrics:")
        for key, val in metrics.items():
            print(f"  {key}: {val.item():.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
