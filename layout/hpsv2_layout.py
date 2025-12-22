import os, json, re
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="")

SYSTEM_PROMPT = """
You are an expert layout planner for multi-instance text-to-image generation.
Given a caption, propose a plausible, aesthetically balanced spatial layout for all mentioned objects.

**Core Logic for Spatial Arrangement:**
1. **Analyze Relations:** First, identify spatial prepositions (e.g., "behind", "in front of", "holding", "sitting on").
2. **Handle Occlusion:**
   - If a relation implies **occlusion** (e.g., "A dog behind a teddy bear"), you **MUST generate overlapping bounding boxes** to reflect depth. The occluded object (background) need to intersect with the occluding object (foreground).
   - If objects are independent (e.g., "A cat and a dog"), avoid overlap to ensure object distinctness.
3. **Safety Margin:** For non-overlapping objects, maintain a small gap (approx. 2-4%) between boxes and from image borders to prevent "concept bleeding" (visual merging of unrelated objects).

Output requirements:
- Return JSON only (no prose).
- For each object, output a bbox [xmin, ymin, xmax, ymax] ∈ [0,1].
- Use caption wording for phrases.
- **Background:** Specify a full-image or large bbox for the background context if needed.

Size & composition priors:
- Vehicles/Furniture: Large (25–45%), bottom-anchored.
- Animals/People: Medium (12–25%).
- Small items: Small (2–10%).
- **Depth Priority:** For "behind" relations, the background object is usually positioned slightly higher or centrally aligned with the foreground object.

Schema:
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

def _extract_json_block(text: str):
    if not text:
        raise ValueError("empty content")
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        cand = "".join(parts[1:-1]).strip()
        if cand.lower().startswith("json"):
            cand = cand[4:].lstrip()
        s = cand or s

    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)

    s = s.strip()
    if not s:
        raise ValueError("no json object found")

    return json.loads(s)

def _try_parse(text: str):
    if not text or not text.strip():
        return None
    try:
        return _extract_json_block(text)
    except Exception:
        try:
            return json.loads(text) 
        except Exception:
            return None

def _fallback_layout(caption: str, idx: int):
    return {
        "index": idx,
        "prompt": caption or "",
        "instances": [{"phrase": "scene", "bbox_xyxy": [0.02, 0.02, 0.98, 0.98]}],
        "bg_prompt": "",
        "neg": "",
        "_warning": "fallback_generated_due_to_parse_error"
    }

def _chat_create(client, *, model, system_prompt, user_prompt, max_tokens=300, temperature=0.2, use_json_mode=True):
    args = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if use_json_mode:
        try:
            return client.chat.completions.create(
                **args, response_format={"type": "json_object"}
            )
        except Exception:
            pass
    return client.chat.completions.create(**args)

def generate_layout(client, caption, model="gpt-4o-2024-05-13", max_retries=2):
    user_prompt = f"Image caption: {caption}"
    content_last = None

    for attempt in range(max_retries + 1):
        resp = _chat_create(
            client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=300,
            temperature=0.2 if attempt == 0 else 0.0,
            use_json_mode=True,
        )
        content = (resp.choices[0].message.content or "").strip()
        content_last = content
        parsed = _try_parse(content)
        if parsed is not None:
            return parsed

        user_prompt = (
            f"Image caption: {caption}\n"
            "Return **JSON only** (no explanations, no markdown, no code fences). "
            "Use double quotes for keys/strings. Ensure valid commas and arrays of four floats for bbox_xyxy."
        )

    fb = _fallback_layout(caption, idx=-1)
    fb["_raw"] = content_last
    return fb

def load_prompts_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}는 리스트 형식이 아님")

    prompts = []
    for x in data:
        if isinstance(x, dict):
            p = x.get("prompt") or x.get("caption") or x.get("text") or ""
        else:
            p = str(x)
        p = p.strip()
        if p:
            prompts.append(p)
    return prompts

def main(data_dir, out="layouts.json"):
    styles = ["anime", "concept-art", "paintings", "photo"]

    aggregated = []
    idx = 0
    for style in styles:
        path = os.path.join(data_dir, f"{style}.json")
        prompts = load_prompts_from_json(path)
        for cap in tqdm(prompts, desc=f"{style}", unit="cap"):
            obj = generate_layout(client, cap, model="gpt-4o-2024-05-13", max_retries=2)
            obj.setdefault("prompt", cap)
            obj["style"] = style
            obj["index"] = idx
            aggregated.append(obj)
            idx += 1

    with open(out, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(aggregated)} layouts to {out}")

if __name__ == "__main__":
    main("../data/hpd", out="../data/layouts_hpd.json")
