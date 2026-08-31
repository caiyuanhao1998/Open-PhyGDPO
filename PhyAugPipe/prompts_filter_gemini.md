# VLM Correction & Re-evaluation Instructions

For **each video** (4 key frames + one JSON description), follow the **five-step pipeline** below.  
Respond **once per video** with **exactly one single-line JSON object** and nothing else.

---

## Input Components
1. **Video Frames** — RGB video frames from the same clip, supplied via `<|video_pad|>`.
2. **JSON Description (text)** — fields:
   - `original`, `parse`, `reason`, `extended`

---

## Output Fields  
Return a JSON object with the same keys, fully corrected / regenerated:

| Key | Required Content |
|-----|------------------|
| `original` | **Unchanged** (copy of input). |
| `parse` | *Corrected* entities / actions / forces / outcomes. |
| `reason` | New 1–2-sentence physical explanation. |
| `extended` | New enriched caption (≤ 100 words) that adds visual physical details found **only in the frames** while keeping original facts. |
| `physics_related_score` | Float ∈ [0, 1]. |
| `physics_label` | 1 if score ≥ 0.60 else 0. |

**Example Output 1**
```json
{
  "original":"A surfer rides a huge wave in deep-blue water.",
  "parse":{
    "entities":["surfer (agent)","wave (water)","ocean (liquid)","sky (background)"],
    "actions":["surfer rides wave","wave curls forward"],
    "forces":["gravity","buoyancy","wave energy"],
    "outcomes":["surfer maintains balance","spray formation"]
  },
  "reason":"Gravity pulls the surfer downward while buoyancy and wave energy propel the board forward, letting the surfer maintain balance on the curling wave.",
  "extended":"The surfer, clad in a white shirt and black shorts, balances on the towering wave as foamy spray arcs behind him. Golden sunlight glints off the glossy board while the deep-blue water heaves under wave energy, driving the surfer forward.",
  "physics_related_score":0.88,
  "physics_label":1
}
```

**Example Output 2**
```json
{
  "original": "In the video, a person is seen building a wall with a sloped track on it in a video game. The person is using a tool to place blocks on the wall, creating a sloped track. The track is made of white blocks, and the wall is made of grey blocks. The person is standing on the ground, and the sky is visible in the background.",
  "parse": {
    "entities": [
      "person (agent)",
      "tool (object)",
      "blocks (material)",
      "wall (structure)",
      "track (structure)",
      "ground (surface)",
      "sky (environment)"
    ],
    "actions": [
      "person places blocks",
      "person builds track",
      "person uses tool"
    ],
    "forces": [
      "gravity",
      "friction",
      "compression"
    ],
    "outcomes": [
      "track construction",
      "wall construction"
    ]
  },
  "reason": "The person uses the tool to place blocks, overcoming friction and gravity to stack them into a sloped track and wall structure in a video game. The blocks are arranged to form a functional design under the person's guidance.",
  "extended": "The person uses the tool to place white blocks, overcoming friction and gravity to stack them into a sloped track and wall structure under bright sunlight in a video game. The grey wall contrasts with the white track, and the scene is captured in a wide-angle shot with the sky visible in the background.",
  "physics_related_score":0.05,
  "physics_label":0
}
```


---

## Step 1 – Verify & Fix `parse`
* Compare `parse` with both **video content** and `original` prompt.
* Ensure:
  - Each **entity** is visible / implied (e.g., surfer, wave, ocean).
  - **Actions** match visible motion (surfing, bouncing, pouring, etc.).
  - **Forces** are physically plausible (gravity, buoyancy, impact, drag, etc.).
  - **Outcomes** describe observable results (splash, deformation, rebound, etc.).
* Remove hallucinated items; add missing ones.

---

## Step 2 – Camera-Motion Filter  
If camera motion (pan/tilt/zoom, first-person sway) dominates the scene **and** no clear physical interaction between entities occurs, later **cap** `physics_related_score` at 0.10 and set `physics_label` = 0.

---

## Step 3 – Realism Check  
Inspect frames for obvious non-realistic styles:
- animation, cartoon, anime, pixel art, video game, 3-D game engine, Lego stop-motion, etc.
If detected, cap score ≤ 0.10 and set label = 0.

---

## Step 4 – Re-Reason (`reason`)
*Write 1–2 concise sentences*:
- Mention every **entity** (exact words, no omissions).
- Specify at least one **force → outcome** causal link.
- Avoid vague phrases (“some physics”, “clearly”) and non-visual senses (sound, smell).

---

## Step 5 – Re-Extend (`extended`)
Create an enriched caption that:
1. **Retains** wording and details in the `original` prompts as much as possible.
2. **Highlights** causal physics from Step 4 (cause → effect).
3. **Adds ≤ 3 new physical visual details** **seen in frames but absent from `original`**, choosing from:
   - lighting / color / texture  
   - shadow / reflection  
   - deformation traces  
   - fine particles (spray, dust, sparks)  
   - kinematic detail (speed, rebound height)  
   - ambient environment (fog, drizzle, breeze)
4. Total ≤ 100 words, no new objects or auditory content.

---

## Step 6 - Scoring

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


**Important** – Scores must not be influenced by stylistic language, sensory embellishments, or information absent from previous steps

---