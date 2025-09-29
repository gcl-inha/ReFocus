import argparse
import json
import os
import re
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from migc.migc_utils import seed_everything
from diffusers import EulerDiscreteScheduler, AutoPipelineForImage2Image
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
from migc.migc_utils import load_migc
from transformers import CLIPModel, CLIPProcessor

torch.set_grad_enabled(False)

DEFAULT_QUALITY_PREFIX = "masterpiece, best quality, "
DEFAULT_NEG = "worst quality, low quality, bad anatomy, watermark, text, blurry"

class CLIPScorer():
    def __init__(self, model_id="openai/clip-vit-large-patch14", dtype=torch.float32, device=None):
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.clip.eval()
        self.clip = self.clip.to(self.device)

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps=None):
        device = self.device
        if isinstance(images, torch.Tensor) and images.dtype == torch.float32 and images.max() <= 1.0:
            do_rescale = False
        else:
            do_rescale = True
        image_inputs = self.processor(images=images, return_tensors="pt", do_rescale=do_rescale)
        image_inputs = {k: v.to(self.dtype).to(device) for k, v in image_inputs.items()}
        image_embeds = self.clip.get_image_features(**image_inputs)
        image_embeds = image_embeds / torch.linalg.vector_norm(image_embeds, dim=-1, keepdim=True)

        if prompts is None:
            return torch.zeros(image_embeds.shape[0], device=device)

        if not isinstance(prompts, list):
            prompts = [prompts] * image_embeds.shape[0]
        elif len(prompts) == 1 and image_embeds.shape[0] > 1:
            prompts = prompts * image_embeds.shape[0]

        text_inputs = self.processor.tokenizer(
            prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        text_embeds = self.clip.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.linalg.vector_norm(text_embeds, dim=-1, keepdim=True)

        image_embeds = image_embeds.to(self.dtype)
        text_embeds = text_embeds.to(self.dtype)
        similarities = torch.sum(image_embeds * text_embeds, dim=1)
        return similarities


def build_refine_prompt(
    caption: str,
    phrases=None,
    use_phrases: bool = False,
    quality_prefix: str | None = None
) -> str:
    if quality_prefix is None:
        quality_prefix = DEFAULT_QUALITY_PREFIX
    base = f"{quality_prefix}{caption}".strip().rstrip(",")
    if use_phrases and phrases:
        phrase_str = ", ".join(p.strip() for p in phrases if p and p.strip())
        if phrase_str:
            return f"{base}, {phrase_str}"
    return base

@torch.no_grad()
def refine_with_img2img(
    img2img_pipe: AutoPipelineForImage2Image = None,
    init_pil: Image.Image = None,
    prompt: str = None,
    negative_prompt: str | None = None,
    steps: int = None,
    strength: float = None,
    guidance_scale: float = None,
    seed: int | None = None,
    device: torch.device = None,
) -> Image.Image:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    result = img2img_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_pil.convert("RGB"),
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    return result

def parse_args():
    p = argparse.ArgumentParser(description="Generate images grouped by style: <outdir>/<style>/<00000>.jpg")
    p.add_argument("layouts_file", type=str, help="Path to layouts JSON file (list).")
    p.add_argument("--outdir", type=str, default="outputs_by_style", help="Root dir to write results to")

    p.add_argument("--sd1x_path", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--migc_ckpt", type=str, default="pretrained_weights/MIGC_SD14.ckpt")
    p.add_argument("--steps", type=int, default=50, help="number of sampling steps")
    p.add_argument("--MIGCsteps", type=int, default=25, help="Number of MIGC-specific conditioning steps")
    p.add_argument("--quality_prefix", type=str, default=DEFAULT_QUALITY_PREFIX)
    p.add_argument("--default_neg", type=str, default=DEFAULT_NEG)
    p.add_argument("--H", type=int, default=None, help="image height")
    p.add_argument("--W", type=int, default=None, help="image width")
    p.add_argument("--scale", type=float, default=7.5, help="CFG scale")
    p.add_argument("--seed", type=int, default=42, help="seed per item")

    p.add_argument("--n_samples", type=int, default=4, help="per-k candidate samples")
    p.add_argument("--k_candidates", type=int, default=1, help="num of pre-refine candidates")
    p.add_argument("--m_candidates", type=int, default=1, help="FINAL images to save per item")

    p.add_argument("--refine", action="store_true", help="Enable img2img refinement")
    p.add_argument("--use_BON_1", action="store_true", help="CLIP-select among n_samples before refine")
    p.add_argument("--use_BON_2", action="store_true", help="CLIP-select among k_candidates after refine")
    p.add_argument("--img2img_model_id", type=str, default="stabilityai/sdxl-turbo", help="Img2img model id")
    p.add_argument("--img2img_steps", type=int, default=2)
    p.add_argument("--img2img_strength", type=float, default=0.5)
    p.add_argument("--img2img_guidance", type=float, default=0.0)
    p.add_argument("--refine_use_phrases", action="store_true", help="Append phrases to refine prompt")

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

def build_migc_inputs(item: Dict[str, Any], quality_prefix: str, fallback_neg: str):
    caption: str = item["prompt"]
    phrases = [inst["phrase"] for inst in item["instances"]]
    boxes = [inst["bbox_xyxy"] for inst in item["instances"]]

    prompt_global = quality_prefix + caption
    prompt_final = [[prompt_global, *phrases]]
    bboxes = [boxes]
    neg = item.get("neg", fallback_neg)
    return prompt_final, bboxes, neg, caption, phrases

def call_migc(pipe: StableDiffusionMIGCPipeline, call_kwargs: dict, want: int):
    images_collected = []
    remaining = want
    while remaining > 0:
        per_call = min(call_kwargs.get("num_images_per_prompt", 1), remaining)
        call_kwargs["num_images_per_prompt"] = per_call
        try:
            out = pipe(**call_kwargs)
            imgs = out.images
            if not isinstance(imgs, list):
                imgs = [imgs]
        except TypeError:
            imgs = [pipe(**{k: v for k, v in call_kwargs.items() if k != "num_images_per_prompt"}).images[0]]
        images_collected.extend(imgs[:per_call])
        remaining -= per_call
    return images_collected

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

    # MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(opt.sd1x_path)
    pipe.attention_store = AttentionStore()
    load_migc(pipe.unet, pipe.attention_store, opt.migc_ckpt, attn_processor=MIGCProcessor)
    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # CLIP scorer
    clip_scorer = CLIPScorer(device=device, dtype=torch.float32)

    # img2img (optional)
    img2img_pipe = None
    if opt.refine:
        variant = "fp16" if device.type == "cuda" else None
        img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            opt.img2img_model_id,
            torch_dtype=dtype,
            variant=variant
        ).to(device)

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
            seed_everything(opt.seed)
            prompt_final, bboxes, negative_prompt_item, caption, phrases = build_migc_inputs(
                item, opt.quality_prefix, opt.default_neg
            )
            print(f"[{style}] #{counters[style]:05d}  '{caption}'")

            with torch.no_grad():
                for _m in range(max(1, opt.m_candidates)):
                    final_images = []
                    for _k in range(max(1, opt.k_candidates)):
                        all_generated_images = []
                        for _n in range(max(1, opt.n_samples)):
                            call_kwargs = dict(
                                prompt=prompt_final,
                                bboxes=bboxes,
                                num_inference_steps=opt.steps,
                                guidance_scale=opt.scale,
                                MIGCsteps=opt.MIGCsteps,
                                aug_phase_with_and=False,
                                negative_prompt=(opt.negative_prompt or negative_prompt_item),
                                num_images_per_prompt=1,
                            )
                            if opt.H is not None:
                                call_kwargs["height"] = opt.H
                            if opt.W is not None:
                                call_kwargs["width"] = opt.W
                            imgs = call_migc(pipe, call_kwargs, want=1)
                            all_generated_images.extend(imgs)

                        if opt.use_BON_1:
                            scores_bon1 = clip_scorer(all_generated_images, caption)
                            best_idx = torch.argmax(scores_bon1).item()
                            best_image = all_generated_images[best_idx]
                        else:
                            best_image = all_generated_images[0]

                        if opt.refine:
                            refine_prompt = build_refine_prompt(
                                caption=caption,
                                phrases=phrases,
                                use_phrases=opt.refine_use_phrases,
                                quality_prefix=opt.quality_prefix
                            )
                            final_image = refine_with_img2img(
                                img2img_pipe=img2img_pipe,
                                init_pil=best_image,
                                prompt=refine_prompt,
                                negative_prompt=(opt.negative_prompt or negative_prompt_item),
                                steps=opt.img2img_steps,
                                strength=opt.img2img_strength,
                                guidance_scale=opt.img2img_guidance,
                                seed=opt.seed,
                                device=device,
                            )
                        else:
                            final_image = best_image
                        final_images.append(final_image)

                    if opt.use_BON_2 and len(final_images) > 1:
                        scores_bon2 = clip_scorer(final_images, caption)
                        chosen = final_images[torch.argmax(scores_bon2).item()]
                    else:
                        chosen = final_images[0]

                    out_name = f"{counters[style]:05d}.jpg"
                    chosen.convert("RGB").save(os.path.join(style_dir, out_name), format="JPEG", quality=95)
                    counters[style] += 1

    print("Done. Saved per-style under:", opt.outdir)

if __name__ == "__main__":
    main()
