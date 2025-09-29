# COMPOSITIONAL IMAGE SYNTHESIS WITH INFERENCE TIME SCALING
[[**arXiv**]()] [[**Project Page**]()]

[Minsuk Ji](), [Sanghyeok Lee](),  [Namhyuk Ahn](https://nmhkahn.github.io)<sup>†</sup> <br>
Inha University <br>
Corresponding author<sup>†</sup><br>

## Abstract
Despite their impressive realism, modern text-to-image diffusion models still struggle with compositionality, often failing to render accurate object counts, attributes, and spatial relations.
To address this challenge, we present a training-free framework that combines an object-centric approach with self-refinement to improve layout faithfulness while preserving aesthetic quality.
Specifically, we leverage large language models (LLMs) to synthesize explicit layouts from input prompts, and we inject these layouts into the image generation process, where a object-centric vision-language model (VLM) judge re-ranks multiple candidates to select the most prompt-aligned outcome iteratively.
By unifying explicit layout-grounding with self-refine-based inference-time scaling, our framework achieves stronger scene alignment with prompts compared to recent text-to-image models.

## Overview
<img width="1980" height="596" alt="pipeline" src="https://github.com/user-attachments/assets/442467c6-ecd6-4f16-b994-68ad33dab2ed" />
## Visual comparison
## Stage 1
## Stage 2
## Stage 3

## Reproduce Experiments in Table 2
<img width="902" height="195" alt="image" src="https://github.com/user-attachments/assets/f12fa904-3d02-4716-bad6-c1bb460d4a86" />

```shell
## SDXL model
python geneval/img2gen_base.py \
    ../data/geneval/evaluation_metadata.jsonl\
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --outdir ../outputs/base_geneval \
    --n_samples 4 
```

```shell
## SD1.5
python geneval/img2gen_base.py \
    ../data/geneval/evaluation_metadata.jsonl\
    --model stabilityai/stable-diffusion-1.5 \
    --outdir ../outputs/sd1.5_geneval \
    --n_samples 4 
```

```shell
## zizgzag model
python geneval/img2gen_zigzag.py \
    ../data/geneval/evaluation_metadata.jsonl\
    --outdir ../outputs/zigzag_geneval \
    --n_samples 4 
```

```shell
## gligen model
python geneval/img2gen_gligen.py \
    ../data/layouts_geneval.json \
    --use_BON_1 \
    --outdir ../outputs/gligen_geneval \
```

```shell
## ours model
python geneval/img2gen_hybridclip.py \
    ../data/layouts_geneval.json \ 
    --refine \
    --use_BON_1 \
    --use_BON_2 \
    --outdir ../outputs/geneval 
```
After generating images with the above commands, we follow the official Geneval evaluation protocol as described in the [Geneval repository](https://github.com/djghosh13/geneval/tree/main?tab=readme-ov-file#evaluation) to compute all reported metrics.

## Reproduce Experiments in Table 3
<img width="567" height="256" alt="image" src="https://github.com/user-attachments/assets/8c12d234-0675-4d8e-9bc8-c0c93ab9a1b5" />

```shell
## SD1.5
python hpsv2/img2hps_base.py \
    ../data/layouts_hpd.json\
    --model stabilityai/stable-diffusion-1.5 \
    --outdir ../outputs/sd1.5_hpsv2 \
    --n_samples 4 
```

```shell
## SD2.0
python hpsv2/img2hps_base.py \
    ../data/layouts_hpd.json\
    --model stabilityai/stable-diffusion-2 \
    --outdir ../outputs/sd2.0_hpsv2 \
    --n_samples 4 
```

```shell
## Phase 1
python hpsv2/img2hps_ours.py \
    ../data/layouts_hpd.json\
    --outdir ../outputs/Phase1_hpsv2
```

```shell
## Phase 1 + ITS(BoN)
python hpsv2/img2hps_ours.py \
    ../data/layouts_hpd.json\
    --outdir ../outputs/ITS_hpsv2 \
    --use_BON_1
```

```shell
## Phase 1 + ITS(BoN) + Refine(1)
python hpsv2/img2hps_ours.py \
    ../data/layouts_hpd.json\
    --outdir ../outputs/Round1_hpsv2 \
    --refine \
    --use_BON_1
```

```shell
## Phase 1 + ITS(BoN) + Refine(1,2)
python hpsv2/img2hps_ours.py \
    ../data/layouts_hpd.json\
    --outdir ../outputs/Round2_hpsv2 \
    --refine \
    --use_BON_1 \
    --use_BON_2
```
After generating images with the above commands, we follow the official HPSv2 evaluation protocol as described in the [HPSv2 repository](https://github.com/tgxs002/HPSv2?tab=readme-ov-file#evaluating-text-to-image-generative-models-using-hps-v2) to compute all reported metrics.
We then run the evaluation using the provided script:
```shell
python evaluation.py --data-type benchmark --data-path <path_to_generated_images>
```
## Citation
## License
