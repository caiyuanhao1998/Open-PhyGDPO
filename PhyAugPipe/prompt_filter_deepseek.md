# Prompt Extension Instructions (LLM-Driven)

This document defines a 3-step pipeline for processing video-related prompts using an LLM such as GPT-4o. Each step should be handled by the LLM without relying on hardcoded rules. For each input prompt, the output should include:

- `original`: the raw prompt
- `parse`: extracted entities, actions, forces, outcomes
- `reason`: short physical explanation
- `extended`: decorated version enriched with physical visual details
- `physics_related_score`: a numerical score (typically in the range 0–1) that quantifies the degree to which the prompt involves physical interactions, phenomena, or reasoning. A higher score indicates denser and more relevant physical content.
- `physics_label`: a binary or categorical label (e.g., “physics” or “non-physics”) indicating whether the prompt is considered sufficiently physics-related. This can be determined by thresholding physics_related_score or by qualitative reasoning.

---

## Step 1: Parse

**Task**: Given a natural language video prompt, extract the following physical elements:

- **Entities**: physical objects or materials involved (e.g., `"glass (brittle)"`, `"ball (rigid)"`, `"rubber band"`)
- **Actions**: physical behaviors or interactions (e.g., `"object is thrown"`, `"liquid is poured"`, `"collision"`)
- **Forces**: physical forces or mechanisms at play (e.g., `"gravity"`, `"friction"`, `"impact"`, `"compression"`)
- **Outcomes**: observable physical results (e.g., `"bounce"`, `"breakage"`, `"deformation"`, `"reflection"`)

**Guidelines**: 
- Always include every **agent** (person, animal, vehicle) that performs an action as an entity, e.g. `"farmer"`,`"player"`, `"dog"`.
- Never omit the subject that executes the main verb.


**Output format**:
```json
"parse": {
  "entities": [...],
  "actions": [...],
  "forces": [...],
  "outcomes": [...]
}
```

Avoid vague or speculative terms. Only include what is clearly implied by the prompt.

---

## Step 2: Reason

**Task**: Using the parsed elements, explain in 1–2 sentences how the entities interact via physical forces and what results.

**Guidelines**:
- Be specific about forces and outcomes.
- Avoid vague terms like “physics” or “clear result”.
- Use natural and concise language.
- Must include the physical forces or results explicitly (e.g., “thrown with force”, “bounces off the wall”).
- Must repeat all `entities` from Step 1 exactly once (no omissions or substitutions).

**Output format**:
```json
"reason": "..."
```

---

## Step 3: Extend  

**Task**  
Enrich the original prompt by *inserting or highlighting* the physical cause-and-effect relationships identified in **Reason** while preserving the details in the original prompt. The result should remain faithful to the scene—no new objects, people, forces, sounds, or stylistic flourishes may be added.  

**Guidelines**  
- Keep the wording and details of the original prompt intact as much as possible; append or embed a concise causal clause (≤ 100 words total).
- Only reference entities, forces, and outcomes already listed in **Parse**／**Reason**.
- Do **not** add sensory details (light, color, sound, temperature, etc.).
- Use clear, natural English; avoid jargon.
- Strictly forbid non-visual sensory details such as sound, smell, taste, or temperature.

**Output format**  
```json
"extended": "..."
```

---

## Step 4: Score

**Task**  
Please follow the following **Guidelines** and **Interpretation Scale** to create two fields that quantify how “physics-rich” the scene is:

```json
"physics_related_score": 0.00,  // float in [0, 1]
"physics_label":        0       // 1 if score ≥ 0.60, else 0
```

**Guidelines**
- Cross-reference all four sources
  - `original` – the original caption of the scene.
  - `parse` – structured extraction of entities, forces, outcomes.
  - `reason` – explicit causal chain.
  - `extended` – prompt augmented with causal highlights.
  Evaluate consistency among them; if parse or reason omits a force that is clearly present in original or extended, still credit that force.
- Scoring heuristics
  - entities - count distinct physical actors; multi-entity means ≥ 2 that physically interact (touch, collide, push, pull).
  - contact emphasis - Scores ≥ 0.75 require direct contact between ≥ 2 entities that produces a force transfer (e.g., push, collision, deformation).
  - forces and outcomes – require at least one of each to reach 0.60+. Multiple distinct forces or sequential outcomes raise the score.
  - Causal depth – multi-step chains (“A causes B, which causes C”) score higher than single-step.
  - Camera-motion filter - If the prompt explicitly mentions cinematographic movements such as “camera pans”, “camera rotates/tilts/zooms”, subtract ≥ 0.30 from the tentative score. When camera motion is the primary action and no physical interaction occurs between entities, cap `physics_related_score` at 0.20 and set physics_label = 0.
  - Unrealistic-style filter - If the prompt contains obvious style cues such as “cartoon”, “anime”, “pixel art”, "video game", “game footage”, “comic strip”, “manga style”, “sketch”, “line art”, or “Lego stop-motion”: subtract ≥ 0.40 from the tentative `physics_related_score`; if the scene is primarily stylistic (no real-world entities or forces), cap `physics_related_score` at 0.10 and set `physics_label` = 0.
  - Static-aftermath filter - If the prompt only describes the aftermath or static visual state of an event (e.g., damage, destruction, mess) with no visible actions or interactions, subtract ≥ 0.40 from the tentative `physics_related_score`. If no dynamic behavior (e.g., falling, bouncing, pouring) is described in present tense, and all verbs imply past results (e.g., “is damaged”, “is charred”, “has collapsed”), then cap `physics_related_score` at 0.20 and set `physics_label` = 0
- Binary label
  - physics_label = 1 if `physics_related_score` ≥ 0.60; otherwise 0.


**Interpretation Scale**
- **0.90 – 1.00** — Multi-entity interaction with **direct physical contact**, at least **two distinct forces** and **two sequential outcomes**, all coherently reflected across *original*, *reason*, and *extended*; **no** unrealistic-style cues (cartoon, sketch, game footage) and **no** dominant camera motion.
- **0.75 – 0.89** — Either multi-entity contact **or** a complex single entity showing **≥ 2 forces/outcomes**; causal chain clearly reinforced in *extended*; any camera movement is minor background; scene must remain grounded in real-world physics (no stylized fiction).
- **0.60 – 0.74** — Clear single-entity motion with an explicit force **and** outcome; style is realistic, though mild abstraction is allowed; camera movement may be mentioned but is not the focus; no strong cartoon/sketch cues.
- **0.40 – 0.59** — Physics is present but weak or ambiguous, **or** style/setting is partly unrealistic (e.g., sketch-like tone); force **or** outcome may be missing, or camera motion shares the visual focus; aftermath-only scenes (static damage) fall here **before** penalty deductions.
- **0.00 – 0.39** — Scene is dominated by camera motion (e.g., “camera pans”, “camera rotates”) **or** clearly unrealistic/stylized depiction (cartoon, video game, pixel art, Lego animation) **or** purely static aftermath with no ongoing interaction; may include major contradictions, hallucinated forces, or virtually no physical interaction.



Important – Scores must not be influenced by stylistic language, sensory embellishments, or information absent from original, parse, reason, and extended.

---

## Example 1

**Input Prompt**
`A red ball rolls down a wooden ramp and bounces off the floor.`

**Output JSON**
```json
{
  "original": "A red ball rolls down a wooden ramp and bounces off the floor.",
  "parse": {
    "entities": ["red ball (rigid)", "wooden ramp (surface)", "floor (surface)"],
    "actions": ["red ball rolls down ramp", "red ball bounces off floor"],
    "forces": ["gravity", "friction", "normal force"],
    "outcomes": ["acceleration", "elastic bounce"]
  },
  "reason": "Gravity pulls the red ball downward along the wooden ramp, causing it to accelerate. When the red ball hits the floor, it compresses slightly and rebounds upward due to its elasticity.",
  "extended": "Gravity accelerates the red ball down the wooden ramp; on impact with the floor, the ball compresses and rebounds upward because its elasticity converts impact energy into upward motion.",
  "physics_related_score": 0.92,
  "physics_label": 1
}
```

## Example 2

**Input Prompt**
`The video shows a cartoon monkey standing in front of a counter. The monkey is holding a piece of paper and a pencil. The monkey writes on the paper and then shows it to the camera. The monkey appears to be happy and satisfied with what he has written on the paper. The background shows a counter with various items on it, but the focus is on the monkey and the paper he is holding.`

**Output JSON**
```json
{
  "original": "The video shows a cartoon monkey standing in front of a counter. The monkey is holding a piece of paper and a pencil. The monkey writes on the paper and then shows it to the camera. The monkey appears to be happy and satisfied with what he has written on the paper. The background shows a counter with various items on it, but the focus is on the monkey and the paper he is holding.",
  "parse": {
    "entities": [
      "cartoon monkey (agent)",
      "paper (flexible)",
      "pencil (rigid)",
      "counter (surface)"
    ],
    "actions": [
      "monkey holds paper",
      "monkey holds pencil",
      "monkey writes on paper",
      "monkey shows paper to camera"
    ],
    "forces": [
      "mechanical force (pencil on paper)",
      "friction (pencil on paper)",
      "gravity"
    ],
    "outcomes": [
      "writing on paper",
      "monkey's satisfaction"
    ]
  },
  "reason": "The cartoon monkey uses the pencil to apply mechanical force on the paper, resulting in writing. Friction between the pencil and paper allows the mark to form. Gravity keeps the paper and pencil in place during the action.",
  "extended": "The cartoon monkey holds a pencil and paper, using mechanical force to write. Friction between the pencil and paper creates the marks. The monkey then shows the written paper to the camera, satisfied with the result.",
  "physics_related_score": 0.05,
  "physics_label": 0
}
```

## Example 3

**Input Prompt**
`The video shows a broken window and scattered debris on the ground. A chair is lying sideways, and the wall has cracks.`

**Output JSON**
```json
{
  "original": "The video shows a broken window and scattered debris on the ground. A chair is lying sideways, and the wall has cracks.",
  "parse": {
    "entities": ["window (broken)", "debris", "chair", "wall (cracked)"],
    "actions": [],
    "forces": [],
    "outcomes": ["broken window", "scattered debris", "fallen chair", "cracked wall"]
  },
  "reason": "The prompt describes the aftermath of an event, but no physical process or active interaction is observed.",
  "extended": "The scene depicts a static aftermath with no visible forces or movements, only results of prior damage.",
  "physics_related_score": 0.18,
  "physics_label": 0
}
```

---
## Final Output

**You must output exactly one single-line JSON object.  
Do **NOT** add any extra words, explanations, markdown, or line breaks.**