import argparse
import os
import re
import torch

from TR_ToMe import apply_ToMe
from TR_SDTM import apply_SDTM
from diffusers import StableDiffusion3Pipeline


def sanitize_filename(text: str, max_len: int = 60) -> str:
    """Make a caption safe for filenames.
    - Lowercase, replace spaces with underscores
    - Keep alnum and _- only
    - Truncate to max_len
    """
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text[:max_len] if len(text) > max_len else text


def main(args):
    # Device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pipeline
    if args.torch_dtype == "float32":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    elif args.torch_dtype == "float16":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    else:
        raise ValueError("--torch-dtype must be 'float32' or 'float16'")

    pipe = pipe.to(device)

    # Model name -> tag
    model_name = os.path.basename(args.model_path)
    if "stable-diffusion-3-medium-diffusers" in model_name:
        model_name = "SD3M"

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Configure method (no output folder creation; save to script directory)
    tore = args.tore_type
    if tore is None or tore == "Default":
        pass
    elif tore == "ToMe":
        # Optional: tag strings removed since we no longer create subfolders
        apply_ToMe(
            pipe,
            ratio=args.ToMe_ratio,
            sx=args.ToMe_sx,
            sy=args.ToMe_sy,
            use_rand=args.ToMe_use_rand,
            merge_attn=args.ToMe_merge_attn,
            merge_mlp=args.ToMe_merge_mlp,
            pseudo_merge=args.ToMe_pseudo_merge,
        )
    elif tore == "SDTM":
        # Optional: tag strings removed since we no longer create subfolders
        apply_SDTM(
            pipe,
            ratio=args.SDTM_ratio,
            deviation=args.SDTM_deviation,
            switch_step=args.SDTM_switch_step,
            use_rand=args.SDTM_use_rand,
            sx=args.SDTM_sx,
            sy=args.SDTM_sy,
            a_s=args.SDTM_a_s,
            a_d=args.SDTM_a_d,
            a_p=args.SDTM_a_p,
            pseudo_merge=args.SDTM_pseudo_merge,
            mcw=args.SDTM_mcw,
            protect_steps_frequency=args.SDTM_protect_steps_frequency,
            protect_layers_frequency=args.SDTM_protect_layers_frequency,
        )
    else:
        raise ValueError("--tore-type must be one of ['Default', 'ToMe', 'SDTM']")

    # Single caption
    prompt = args.caption
    print(f"Prompt: {prompt}")

    image = pipe(
        prompt=prompt,
        generator=generator,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    ).images[0]

    # Save to a relative output directory under the script folder
    fname = f"{sanitize_filename(prompt)}-seed{args.seed}.jpg"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, args.output_path)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, fname)
    image.save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption",
        type=str,
        default="Ma motorcycle parked on the gravel in front of a garage",
        help="Single text prompt to generate an image",
    )
    parser.add_argument("--model-path", type=str, default="../../checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["float32", "float16"])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tore-type", type=str, choices=["Default", "ToMe", "SDTM"], default="SDTM")
    parser.add_argument("--output-path", type=str, default="samples/demos", help="Relative folder under script dir to save outputs")

    # ToMe args
    parser.add_argument("--ToMe-ratio", type=float, default=0.9)
    parser.add_argument("--ToMe-sx", type=int, default=2)
    parser.add_argument("--ToMe-sy", type=int, default=2)
    parser.add_argument("--ToMe-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ToMe-pseudo-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bind objects together without actual merging.",
    )

    # SDTM args
    parser.add_argument("--SDTM-ratio", type=float, default=0.3)
    parser.add_argument("--SDTM-deviation", type=float, default=0.2)
    parser.add_argument("--SDTM-switch-step", type=int, default=20)
    parser.add_argument("--SDTM-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--SDTM-sx", type=int, default=2)
    parser.add_argument("--SDTM-sy", type=int, default=2)
    parser.add_argument("--SDTM-a-s", type=float, default=0.05)
    parser.add_argument("--SDTM-a-d", type=float, default=0.05)
    parser.add_argument("--SDTM-a-p", type=float, default=2)
    parser.add_argument(
        "--SDTM-pseudo-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bind objects together without actual merging.",
    )
    parser.add_argument(
        "--SDTM-mcw",
        type=float,
        default=0.1,
        help="the weight for merge is w, while for cache is 1-w",
    )
    parser.add_argument(
        "--SDTM-protect-steps-frequency",
        type=int,
        default=3,
        help="Frequency for protecting steps",
    )
    parser.add_argument(
        "--SDTM-protect-layers-frequency",
        type=int,
        default=-1,
        help="Frequency for protecting layers",
    )
    parser.add_argument(
        "--SDTM-cache_each_step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache each step flag (tag only)",
    )

    args = parser.parse_args()
    main(args)
