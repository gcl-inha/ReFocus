import os
from openai import OpenAI
import json, re
from tqdm import tqdm
from pathlib import Path

client = OpenAI(api_key="")

SYSTEM_PROMPT = """
You are a layout planner for multi-instance text-to-image generation.
Given only a caption, you must propose a plausible, aesthetically balanced layout for all objects explicitly mentioned in the caption.

Output requirements:

- For each caption object, output a tight, non-overlapping bounding box in normalized xyxy format [xmin, ymin, xmax, ymax] ∈ [0,1], with xmax>xmin and ymax>ymin.
- Use the caption wording to form a short phrase (e.g., "black bus", "brown cell phone").
- Return JSON only (no prose), following the exact schema shown below.
- Keep a 2–4% margin from the image border when reasonable and avoid awkward overlaps.
- You must also specify the bbox layout for the background.

Size & composition priors (guidelines, not hard rules):

- Vehicles / large furniture (bus, train, sofa): large box (≈ 25–45% of area), often bottom-anchored and spanning more width.
- Animals / people (sheep, dog, person): medium (≈ 12–25%).
- Handhelds / accessories / fruits (cell phone, handbag, banana): small to micro (≈ 2–10%).
- Tall slim objects (umbrella, streetlight): use taller aspect boxes; align to one side.
- For two objects, prefer left–right or foreground(bottom)–background(top) balance.

Schema :
{
  "prompt": "<string>",
  "instances": [
    {"phrase": "<string>", "bbox_xyxy": [<float>, <float>, <float>, <float>]}
  ],
  "bg_prompt": "<string, optional>",
  "neg": "<string, optional>"
}

Output JSON only.
"""

def generate_layout(
    prompt,
    model="gpt-4o-2024-05-13",
    max_tokens=150,
    temperature=0.3,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    use_json_mode=True, 
):
    args = dict(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

    if use_json_mode:
        try:
            return client.chat.completions.create(
                **args,
                response_format={"type": "json_object"},
            )
        except Exception:
            pass
    return client.chat.completions.create(**args)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        candidate = "".join(parts[1:-1]).strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
        text = candidate or text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)

    if not text:
        raise ValueError("Empty content after cleanup")

    return json.loads(text)


def _try_parse_layout(content: str):
    if not content or not content.strip():
        return None
    try:
        return _extract_json(content)
    except Exception:
        try:
            return json.loads(content)
        except Exception:
            return None


def _make_fallback_layout(caption: str, index: int):
    return {
        "index": index,
        "prompt": caption,
        "instances": [
            {"phrase": "scene", "bbox_xyxy": [0.02, 0.02, 0.98, 0.98]}
        ],
        "bg_prompt": "",
        "neg": "",
        "_warning": "fallback_generated_due_to_parse_error",
    }


def run_layouts_from_file(
    in_path: str,
    out_json_path: str = "layouts.json",
    model: str = "gpt-4o-2024-05-13",
    index_start: int = 0,
    file_template: str = "gen_data/{index:05d}.png",
    max_retries: int = 2,
    fallback_on_fail: bool = True,
    **gen_kwargs,
):
    records = load_records(in_path)

    aggregated = []
    for i, rec in enumerate(tqdm(records, desc="Generating layouts", unit="rec"), start=index_start):
        cap = (rec.get("prompt") or rec.get("caption") or rec.get("text") or "").strip()
        user_prompt = f"Image caption: {cap}"

        obj = None
        last_content = None
        for attempt in range(max_retries + 1):
            resp = generate_layout(user_prompt, model=model, **gen_kwargs)
            content = resp.choices[0].message.content if resp and resp.choices else ""
            content = (content or "").strip()
            last_content = content

            parsed = _try_parse_layout(content)
            if parsed is not None:
                obj = parsed
                break

            user_prompt = (
                f"{user_prompt}\n\n"
                "Return JSON only. No prose, no markdown, no code fences."
            )

        if obj is None:
            msg = f"[WARN] JSON parse failed at index {i}. Using fallback."
            if not fallback_on_fail:
                raise ValueError(f"{msg}\nRaw content: {last_content}")
            print(msg)
            obj = _make_fallback_layout(cap, i)
            obj["_raw"] = last_content

        obj["index"] = i
        if not obj.get("prompt"):
            obj["prompt"] = cap
        file_val = file_template.format(index=i, i=i)
        obj.setdefault("file", file_val)

        for k, v in rec.items():
            if k in {"prompt", "index"}:
                continue
            obj.setdefault(k, v)

        aggregated.append(obj)

    with open(out_json_path, "w", encoding="utf-8") as wf:
        json.dump(aggregated, wf, ensure_ascii=False, indent=2)

    print(f"Saved {len(aggregated)} items to {out_json_path}")
    return aggregated

def load_records(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        lower = path.lower()
        if lower.endswith(".jsonl"):
            records = []
            for line_no, ln in enumerate(f, 1):
                s = ln.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSONL parse error at line {line_no}: {e}")
                    continue
                records.append(rec)
            return records

        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error in {path}: {e}")

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for k in ("records", "items", "results", "data"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            if all(isinstance(v, dict) for v in data.values()):
                out = []
                for k, v in data.items():
                    rec = {"id": k}
                    rec.update(v)
                    out.append(rec)
                return out
            if "prompt" in data:
                return [data]

        raise ValueError(f"Unsupported JSON structure in {path}")

def run_layouts_from_jsonl(jsonl_path: str, **kwargs):
    return run_layouts_from_file(jsonl_path, **kwargs)

if __name__ == "__main__":
    outs = run_layouts_from_file(
        "../data/geneval/evaluation_metadata.jsonl",
        out_json_path="../data/layouts_geneval.json",
        model="gpt-4o-2024-05-13",
        max_tokens=300,
        temperature=0.2,
    )
