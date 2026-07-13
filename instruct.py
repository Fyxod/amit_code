#!/usr/bin/env python
"""
InstructPix2Pix White-Box Backend Script.

This script provides a CLI interface to the InstructPix2Pix white-box backend
from the GLASS project. It allows running image edits using the InstructPix2Pix
model with configurable parameters.

Usage:
    python instruct.py -i input.jpg -o output.jpg -p "Make him smile"
    python instruct.py -i input.jpg -o output.jpg -p "Add glasses" --steps 30 --guidance 8.0
"""
import os
import argparse
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Any


@dataclass
class InstructSettings:
    model_id: str = "timbrooks/instruct-pix2pix"
    torch_dtype: str = "float16"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    objective_timestep_index: int = 6
    seed: int = 1234


def save_image(tensor: torch.Tensor, path: str):
    """Save a tensor (C, H, W) in [0, 1] range as PNG."""
    from torchvision.utils import save_image as tv_save
    tv_save(tensor.unsqueeze(0), path)


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


class InstructBackend:
    name = "instruct"

    def __init__(self, device: torch.device, settings: InstructSettings | None = None) -> None:
        self.device = device
        self.settings = settings or InstructSettings()
        self.pipe = self._load()

    def _load(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        if self.device.type != "cuda":
            raise RuntimeError("InstructPix2Pix white-box GLASS runs require CUDA.")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=_dtype(self.settings.torch_dtype),
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for module_name in ("vae", "text_encoder", "unet"):
            module = getattr(pipe, module_name, None)
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        return pipe

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        return self.pipe._encode_prompt(prompt, self.device, 1, False)

    def encode_image_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image = (image_tensor * 2.0 - 1.0).to(device=self.device, dtype=self.pipe.vae.dtype)
        latent = self.pipe.vae.encode(image).latent_dist.mode()
        return latent.to(dtype=self.pipe.unet.dtype)

    def _unet_prediction(self, image_latent: torch.Tensor, embedding: torch.Tensor, reference: dict[str, Any]) -> torch.Tensor:
        noisy = self.pipe.scheduler.scale_model_input(reference["fixed_noise"], reference["timestep"])
        sample = torch.cat([noisy.to(dtype=self.pipe.unet.dtype), image_latent.to(dtype=self.pipe.unet.dtype)], dim=1)
        return self.pipe.unet(
            sample,
            reference["timestep"],
            encoder_hidden_states=embedding,
            return_dict=False,
        )[0]

    def prepare_reference(self, original_tensor: torch.Tensor, prompt: str, objective: str) -> dict[str, Any]:
        with torch.no_grad():
            prompt_embedding = self._encode_prompt(prompt).detach()
            original_latent = self.encode_image_latent(original_tensor).detach()
            self.pipe.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
            steps = self.pipe.scheduler.timesteps
            timestep = steps[min(max(0, self.settings.objective_timestep_index), len(steps) - 1)]
            generator = torch.Generator(device=self.device).manual_seed(self.settings.seed)
            fixed_noise = torch.randn(
                original_latent.shape,
                generator=generator,
                device=self.device,
                dtype=self.pipe.unet.dtype,
            ) * self.pipe.scheduler.init_noise_sigma
            placeholder = {
                "prompt": prompt,
                "prompt_embedding": prompt_embedding,
                "original_latent": original_latent,
                "fixed_noise": fixed_noise,
                "timestep": timestep,
            }
            clean_prediction = self._unet_prediction(original_latent, prompt_embedding, placeholder).detach()
        placeholder["clean_prediction"] = clean_prediction
        placeholder["objective"] = objective
        return placeholder

    def internal_objective(self, perturbed: torch.Tensor, reference: dict[str, Any], objective: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch.nn.functional as F
        perturbed_latent = self.encode_image_latent(perturbed)
        prediction = self._unet_prediction(perturbed_latent, reference["prompt_embedding"], reference)
        vae_mse = F.mse_loss(perturbed_latent.float(), reference["original_latent"].float())
        unet_mse = F.mse_loss(prediction.float(), reference["clean_prediction"].float())
        if objective == "vae_conditioning":
            value = vae_mse
        elif objective == "unet_prediction":
            value = unet_mse
        else:
            raise ValueError(f"Unsupported InstructPix2Pix objective: {objective}")
        return value, {
            "vae_conditioning_mse": vae_mse,
            "unet_prediction_mse": unet_mse,
        }

    @torch.inference_mode()
    def generate_edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=self.settings.num_inference_steps,
            guidance_scale=self.settings.guidance_scale,
            image_guidance_scale=self.settings.image_guidance_scale,
            generator=generator,
        )
        return result.images[0].convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="InstructPix2Pix white-box backend for image editing"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path to save output image")
    parser.add_argument("--prompt", "-p", type=str, default="Make him smile",
                        help="Text prompt for the edit")
    parser.add_argument("--model-id", type=str, default="timbrooks/instruct-pix2pix",
                        help="HuggingFace model ID")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Torch dtype for model")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Text guidance scale")
    parser.add_argument("--image-guidance-scale", type=float, default=1.5,
                        help="Image guidance scale")
    parser.add_argument("--objective-timestep-index", type=int, default=6,
                        help="Timestep index for objective computation")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run on")
    parser.add_argument("--objective", type=str, default=None,
                        choices=["vae_conditioning", "unet_prediction"],
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
    settings = InstructSettings(
        model_id=args.model_id,
        torch_dtype=args.dtype,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        objective_timestep_index=args.objective_timestep_index,
        seed=args.seed,
    )

    print(f"\nInitialising InstructPix2Pix backend...")
    print(f"  Model: {settings.model_id}")
    print(f"  Steps: {settings.num_inference_steps}")
    print(f"  Guidance: {settings.guidance_scale}")
    print(f"  Image Guidance: {settings.image_guidance_scale}")
    print(f"  Dtype: {settings.torch_dtype}")

    backend = InstructBackend(device, settings)

    # Run edit or compute objective
    if args.objective is None:
        # Standard image editing
        print(f"\nRunning image edit with prompt: '{args.prompt}'")
        result = backend.generate_edit(image, args.prompt, args.seed)
        output_dir = os.path.dirname(args.output) or "."
        os.makedirs(output_dir, exist_ok=True)
        result.save(args.output)
        print(f"\nSaved edited image → {args.output}")

        # ── Save difference images ─────
        import torch.nn.functional as F
        from torchvision.transforms.functional import resize
        result_tensor = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Resize result to match original if needed
        if result_tensor.shape[1:] != image_tensor.shape[1:]:
            result_tensor = resize(result_tensor, image_tensor.shape[1:])
        
        # Raw difference (result - original)
        diff = result_tensor - image_tensor  # Difference in [-1, 1]
        diff_vis = (diff + 1.0) / 2.0  # Shift to [0, 1] range (0.5 = no difference)
        diff_vis = diff_vis.clamp(0, 1)
        diff_output = os.path.join(output_dir, "difference.png")
        save_image(diff_vis, diff_output)
        print(f"Saved raw difference image → {diff_output}")

        # 8x enhanced difference
        diff_enhanced = diff * 8.0
        diff_enhanced_vis = (diff_enhanced + 1.0) / 2.0
        diff_enhanced_vis = diff_enhanced_vis.clamp(0, 1)
        diff_enhanced_output = os.path.join(output_dir, "difference_8.png")
        save_image(diff_enhanced_vis, diff_enhanced_output)
        print(f"Saved 8x enhanced difference image → {diff_enhanced_output}")
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
