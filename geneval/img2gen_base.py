import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, StableDiffusionPipeline

from scheduler.scheduling_ddim import DDIMScheduler

from transformers import CLIPModel, CLIPProcessor


torch.set_grad_enabled(False)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="Input prompts as .jsonl (one JSON per line) or .json (list/dict)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="prompthero/openjourney",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument(
        "--k_samples",
        type=int,
        default=1,
        help="number of samples",
    )

    opt = parser.parse_args()
    return opt


# -------- NEW: unified loader for .json / .jsonl --------
def _normalize_list_of_metadatas(items):
    """Coerce items into list[dict] with 'prompt' key."""
    out = []
    for x in items:
        if isinstance(x, str):
            out.append({"prompt": x})
            continue
        if isinstance(x, dict):
            if "prompt" not in x:
                if "caption" in x:
                    x["prompt"] = x["caption"]
                elif "text" in x:
                    x["prompt"] = x["text"]
            if "prompt" in x and isinstance(x["prompt"], str) and x["prompt"].strip():
                out.append(x)
            # silently skip if no prompt
    return out

def load_metadatas(path):
    ext = os.path.splitext(path)[1].lower()
    # Try JSONL first when extension hints it
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        data = []
        for ln in lines:
            data.append(json.loads(ln))
        return _normalize_list_of_metadatas(data)
    # JSON (list or dict wrapper)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return _normalize_list_of_metadatas(obj)
        if isinstance(obj, dict):
            # common wrappers
            for k in ("data", "items", "metadatas"):
                if k in obj and isinstance(obj[k], list):
                    return _normalize_list_of_metadatas(obj[k])
            # dict of {id: {...}}
            if all(isinstance(v, dict) for v in obj.values()):
                return _normalize_list_of_metadatas(list(obj.values()))
        raise ValueError("Unsupported JSON structure: expected list or dict with list under 'data'/'items'/'metadatas'.")
    # Fallback: try JSONL first, then JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        data = [json.loads(ln) for ln in lines]
        return _normalize_list_of_metadatas(data)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return _normalize_list_of_metadatas(obj)
        if isinstance(obj, dict):
            for k in ("data", "items", "metadatas"):
                if k in obj and isinstance(obj[k], list):
                    return _normalize_list_of_metadatas(obj[k])
        raise


def main(opt):
    # Load prompts (now supports .jsonl and .json)
    metadatas = load_metadatas(opt.metadata_file)
    # Load model
    if opt.model == "stabilityai/stable-diffusion-xl-base-1.0":
        model = DiffusionPipeline.from_pretrained(opt.model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        model.enable_xformers_memory_efficient_attention()
    else:
        model = StableDiffusionPipeline.from_pretrained(opt.model, torch_dtype=torch.float16)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.enable_attention_slicing()
    clip_scorer = CLIPScorer(device=device, dtype=torch.float32)

    
    

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            final_samples = list()
            for i in range(opt.k_samples):
                all_generated_images = []
                for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                    # Generate images
                    
                    samples = model(
                        prompt,
                        height=opt.H,
                        width=opt.W,
                        num_inference_steps=opt.steps,
                        guidance_scale=opt.scale,
                        num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                        negative_prompt=opt.negative_prompt or None
                    ).images
                    all_generated_images.extend(samples)
                        
                # # CLIP scoring and best image selection
                # clip_scores = clip_scorer(all_generated_images, prompt)
                # best_idx = torch.argmax(clip_scores).item()
                # best_sample = all_generated_images[best_idx]
                # final_samples.append(best_sample)
                final_samples.extend(all_generated_images)

            for sample in final_samples:
                sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                sample_count += 1
            if not opt.skip_grid:
                all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
