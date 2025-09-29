import argparse
import json
import os
import re
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import DiffusionPipeline, EulerDiscreteScheduler

torch.set_grad_enabled(False)

DEFAULT_QUALITY_PREFIX = "masterpiece, best quality, "
DEFAULT_NEG = "worst quality, low quality, bad anatomy, watermark, text, blurry"

def parse_args():
    p = argparse.ArgumentParser(description="Generate images grouped by style: <outdir>/<style>/<00000>.jpg")
    p.add_argument("layouts_file", type=str, help="Path to layouts JSON file (list).")
    p.add_argument("--outdir", type=str, default="outputs_by_style", help="Root dir to write results to")
    p.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model path")
    p.add_argument("--steps", type=int, default=50, help="number of sampling steps")
    p.add_argument("--quality_prefix", type=str, default=DEFAULT_QUALITY_PREFIX)
    p.add_argument("--default_neg", type=str, default=DEFAULT_NEG)
    p.add_argument("--H", type=int, default=1024, help="image height")
    p.add_argument("--W", type=int, default=1024, help="image width")
    p.add_argument("--scale", type=float, default=7.5, help="CFG scale")
    p.add_argument("--seed", type=int, default=42, help="seed per item")
    p.add_argument("--n_samples", type=int, default=1, help="number of images to generate per item")

    p.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const=("ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
               "extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, "
               "cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"),
        default=None,
        help="negative prompt for guidance"
    )
    return p.parse_args()

def load_layouts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("Invalid layouts file format. Provide a JSON array of layout items.")

def build_sdxl_prompt(item: Dict[str, Any], quality_prefix: str, fallback_neg: str):
    caption: str = item["prompt"]
    phrases = [inst["phrase"] for inst in item["instances"]]
    
    full_prompt = quality_prefix + caption
    if phrases:
        phrase_str = ", ".join(p.strip() for p in phrases if p and p.strip())
        if phrase_str:
            full_prompt = f"{full_prompt}, {phrase_str}"
    
    neg = item.get("neg", fallback_neg)
    return full_prompt, neg, caption

def _norm_style_name(s: str) -> str:
    ss = (s or "unknown").strip().lower()
    ss = re.sub(r"\s+", "-", ss)
    ss = ss.replace("_", "-")
    return ss

def main():
    opt = parse_args()
    os.makedirs(opt.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        opt.model, 
        torch_dtype=dtype, 
        use_safetensors=True, 
        variant="fp16" if device.type == "cuda" else None
    )
    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    items = load_layouts(opt.layouts_file)
    groups: Dict[str, list] = {}
    for seq, it in enumerate(items):
        style = _norm_style_name(str(it.get("style", "unknown")))
        idx_raw = it.get("index", None)
        gidx = None
        if isinstance(idx_raw, int):
            gidx = idx_raw
        else:
            try:
                gidx = int(idx_raw)
            except Exception:
                gidx = None
        sort_key = (0, gidx, seq) if gidx is not None else (1, seq, seq)  # (flag, global_index, fallback_seq)
        groups.setdefault(style, []).append((sort_key, it))

    preferred = ["anime", "concept-art", "paintings", "photo"]
    styles_in_data = list(groups.keys())
    ordered_styles = [s for s in preferred if s in styles_in_data] + [s for s in sorted(styles_in_data) if s not in preferred]
    counters: Dict[str, int] = {}
    for st in ordered_styles:
        os.makedirs(os.path.join(opt.outdir, st), exist_ok=True)
        counters[st] = 0
    for style in ordered_styles:
        entries = sorted(groups[style], key=lambda x: x[0])
        style_dir = os.path.join(opt.outdir, style)

        for _, item in entries:
            torch.manual_seed(opt.seed)
            full_prompt, negative_prompt_item, caption = build_sdxl_prompt(
                item, opt.quality_prefix, opt.default_neg
            )
            print(f"[{style}] #{counters[style]:05d}  '{caption}'")

            with torch.no_grad():
                for sample_idx in range(opt.n_samples):
                    generator = torch.Generator(device=device).manual_seed(opt.seed + sample_idx)
                    
                    result = pipe(
                        prompt=full_prompt,
                        negative_prompt=(opt.negative_prompt or negative_prompt_item),
                        num_inference_steps=opt.steps,
                        guidance_scale=opt.scale,
                        height=opt.H,
                        width=opt.W,
                        generator=generator,
                        num_images_per_prompt=1,
                    )
                    
                    generated_image = result.images[0]
                    
                    if opt.n_samples == 1:
                        out_name = f"{counters[style]:05d}.jpg"
                    else:
                        out_name = f"{counters[style]:05d}_{sample_idx:02d}.jpg"
                    
                    generated_image.convert("RGB").save(os.path.join(style_dir, out_name), format="JPEG", quality=95)
                
                counters[style] += 1

    print("Done. Saved per-style under:", opt.outdir)

if __name__ == "__main__":
    main()
