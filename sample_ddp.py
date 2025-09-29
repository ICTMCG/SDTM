import json
import argparse
import torch
import torch.distributed as dist
import os

from tqdm import tqdm
from TR_ToMe import apply_ToMe
from TR_SDTM import apply_SDTM
from diffusers import StableDiffusion3Pipeline


def load_captions(file_path):
    """
    Reads the 'annotations' section of a specified JSON file, extracts the image_id and caption from each annotation,
    and returns a list containing this data, ensuring that each image_id is unique and has the longest caption.

    :param file_path: Path to the JSON file
    :return: A list where each element is a dictionary containing 'image_id' and the longest 'caption' for that image_id
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    annotations = data
    # Dictionary to store the longest caption for each image_id
    longest_captions = {}

    # Iterate through each annotation
    for item in annotations:
        image_id = item['image_id']
        caption = item['caption']
        # If the image_id is already in the dictionary and the current caption is longer, update it
        if image_id in longest_captions:
            if len(caption) > len(longest_captions[image_id]['caption']):
                longest_captions[image_id] = {'image_id': image_id, 'caption': caption}
        else:
            # Otherwise, add the image_id and caption to the dictionary
            longest_captions[image_id] = {'image_id': image_id, 'caption': caption}

    # Extract values from the dictionary to form the final list
    image_captions = list(longest_captions.values())

    return image_captions

def main(args):
    """Run sampling with DDP, mirroring single-GPU sample.py behavior."""
    # Initialize the process group (NCCL for NVIDIA GPUs)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load and deduplicate captions (rank0 writes back once)
    file_path = args.caption_path
    captions_list = load_captions(file_path)
    if rank == 0:
        with open(args.caption_path, 'w', encoding='utf-8') as f:
            json.dump(captions_list, f, ensure_ascii=False, indent=4)
        print(f"Loaded unduplicated {len(captions_list)} captions.")
    dist.barrier()

    # Load the pipeline
    if args.torch_dtype == "float32":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    elif args.torch_dtype == "float16":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    else:
        raise ValueError("--torch-dtype must be 'float32' or 'float16'")
    pipe = pipe.to(device)

    # Ensure all processes have loaded the model before proceeding
    dist.barrier()

    # Output path and method configuration
    model_name = os.path.basename(args.model_path)
    if "stable-diffusion-3-medium-diffusers" in model_name:
        model_name = "SD3M"

    tore = args.tore_type
    if tore is None or tore in ["Default", "default"]:
        output_path = os.path.join(
            args.output_path,
            f"{model_name}-Default-{args.height}x{args.width}-steps{args.num_inference_steps}-cfg{args.guidance_scale}-seed{args.seed}"
        )
    elif tore == "ToMe":
        output_path = os.path.join(
            args.output_path,
            f"{model_name}-ToMe-pseudo_merge{args.ToMe_pseudo_merge}-{args.ToMe_ratio}-{args.ToMe_sx}x{args.ToMe_sy}-"
            f"MergeAttn{args.ToMe_merge_attn}-MergeMLP{args.ToMe_merge_mlp}-"
            f"{args.height}x{args.width}-steps{args.num_inference_steps}-cfg{args.guidance_scale}-seed{args.seed}"
        )
        output_path = (
            output_path
            .replace("pseudo_mergeFalse", "Merge").replace("pseudo_mergeTrue", "PseudoMerge")
            .replace("MergeAttnTrue", "MergeAttn").replace("MergeAttnFalse", "UnMergeAttn")
            .replace("MergeMLPTrue", "MergeMLP").replace("MergeMLPFalse", "UnMergeMLP")
        )
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
        sdtm_tag = (
            f"{model_name}-SDTM-"
            f"R{args.SDTM_ratio:g}-D{args.SDTM_deviation:g}-Sw{args.SDTM_switch_step}-"
            f"rnd{int(args.SDTM_use_rand)}-{args.SDTM_sx}x{args.SDTM_sy}-"
            f"as{args.SDTM_a_s:g}-ad{args.SDTM_a_d:g}-ap{args.SDTM_a_p:g}-"
            f"Pm{'PM' if args.SDTM_pseudo_merge else 'M'}-W{args.SDTM_mcw:g}-"
            f"Ps{args.SDTM_protect_steps_frequency}-Pl{args.SDTM_protect_layers_frequency}-CES{args.SDTM_cache_each_step}"
        )
        output_path = os.path.join(
            args.output_path,
            f"{sdtm_tag}-{args.height}x{args.width}-steps{args.num_inference_steps}-cfg{args.guidance_scale}-seed{args.seed}"
        )
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

    # Rank-0 creates output directory
    if rank == 0 and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    dist.barrier()

    # Split data among ranks
    batch_size = args.batch_size
    num_samples = len(captions_list)
    indices = list(range(num_samples))
    indices_per_rank = indices[rank::world_size]
    local_captions_list = [captions_list[i] for i in indices_per_rank]

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Generate
    for i in tqdm(range(0, len(local_captions_list), batch_size), desc=f"Rank {rank}"):
        batch_captions = local_captions_list[i: i + batch_size]
        prompt_list = [item['caption'] for item in batch_captions]
        id_list = [item['image_id'] for item in batch_captions]

        images = pipe(
            prompt=prompt_list,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images

        for j, image in enumerate(images):
            image_id = str(id_list[j]).zfill(12)
            image.save(os.path.join(output_path, f"{image_id}.jpg"))

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-path", type=str, default='../../datasets/COCO2017/longest_captions.json')
    parser.add_argument("--output-path", type=str, default="samples")
    parser.add_argument("--model-path", type=str, default="../../checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["float32", "float16"])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tore-type", type=str, choices=["Default", "ToMe", "SDTM"], default="SDTM")

    # ToMe args
    parser.add_argument("--ToMe-ratio", type=float, default=0.9)
    parser.add_argument("--ToMe-sx", type=int, default=2)
    parser.add_argument("--ToMe-sy", type=int, default=2)
    parser.add_argument("--ToMe-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ToMe-pseudo-merge", action=argparse.BooleanOptionalAction, default=True,
                        help="Bind objects together without actual merging.")

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
    parser.add_argument("--SDTM-pseudo-merge", action=argparse.BooleanOptionalAction, default=False,
                        help="Bind objects together without actual merging.")
    parser.add_argument("--SDTM-mcw", type=float, default=0.1,
                        help="the weight for merge is w, while for cache is 1-w")
    parser.add_argument("--SDTM-protect-steps-frequency", type=int, default=3,
                        help='Frequency for protecting steps')
    parser.add_argument("--SDTM-protect-layers-frequency", type=int, default=-1,
                        help='Frequency for protecting layers')
    parser.add_argument("--SDTM-cache_each_step", action=argparse.BooleanOptionalAction, default=True,
                        help="Cache each step flag (tag only)")

    args = parser.parse_args()
    main(args)