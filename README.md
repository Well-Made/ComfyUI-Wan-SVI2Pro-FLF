# ComfyUI-Wan-SVI2Pro-FLF

Custom nodes for ComfyUI that combine Wan 2.2 SVI 2 Pro motion continuity with First/Last Frame (FLF) style control over the end of a clip, plus a small utility for trimming Wan video latents.

## Nodes

### Wan SVI 2 Pro FLF

**Class:** `WanImageToVideoSVIProFLF`  
**Category:** `conditioning/video_models`

Combines:

- **Start:** Wan SVI 2 Pro–style motion continuity (like `WanImageToVideoSVIPro`), using:
  - `anchor_samples` as the anchor latent(s) for the current segment.
  - An optional tail from `prev_samples` (last `motion_latent_count` temporal slots) to continue motion between segments.
- **End:** FLF-style control (like `WanFirstLastFrameToVideoLatent` / `FLF2V`):
  - The last temporal slots are hard-locked to `end_samples` using `concat_mask`.

This gives you:

- Strong content anchoring at the beginning of each segment.
- Smooth motion continuity between segments via `prev_samples`.
- Deterministic control over the last frames of each segment via `end_samples`.

**Inputs:**

- `positive` / `negative` – standard ComfyUI conditionings.
- `length` – target video length in frames. Wan video uses stride 4, so this is converted to latent T.
- `prev_samples` – optional latents from the previous segment (`B,C,T,H,W`). The last `motion_latent_count` slots are used as motion tail.
- `anchor_samples` – anchor latent(s) for the current segment (`B,C,T,H,W`), usually the first frame or a short clip.
- `end_samples` – optional target end latent(s) (`B,C,T,H,W`). The last `T_end` slots are copied into the tail of the segment and hard-locked.
- `motion_latent_count` – how many last temporal slots to take from `prev_samples` (0 disables motion continuity; 1–2 is typical).

**Outputs:**

- `positive` / `negative` – conditionings with `concat_latent_image` and `concat_mask` injected.
- `latent` – empty latent (`{"samples": zeros}`) for Wan’s video sampler.

Typical usage:

- **First segment:**  
  - `prev_samples` empty, `motion_latent_count = 0`,  
  - `anchor_samples` = first frame latent, `end_samples` = last frame latent.
- **Subsequent segments:**  
  - `prev_samples` = latents from previous segment (often trimmed with `WanCutLastSlot`),  
  - `anchor_samples` = anchor for the new segment,  
  - `end_samples` = new target last frame,  
  - `motion_latent_count` = 1–2.

---

### Wan Cut Last Slot

**Class:** `WanCutLastSlot`  
**Category:** `latent/video`

Utility node that trims temporal slots from the end of a Wan video latent clip.

In Wan 2.2, one temporal slot corresponds to 4 frames (stride = 4). Cutting 1 slot effectively removes the latent information for the last ~4 frames.

The main motivation is to avoid a conflict between **hard‑locked last frames** (FLF logic) and **SVI Pro motion continuity**. When the very last slot is fully fixed by `end_samples`, SVI still tries to “steer” the motion trajectory through that slot. These two behaviors can clash and produce heavy artifacts or “crumpled” frames exactly at the hard‑locked end. The simplest and most robust workaround is to drop that last temporal slot entirely before using the segment as `prev_samples` for the next one.

**Inputs:**

- `latents` – Wan video latents (`{"samples": tensor[B,C,T,H,W]}`).
- `slots_to_cut` – how many last temporal slots to cut (T dimension). At least one slot is always preserved internally.

**Output:**

- `trimmed_latents` – same structure as input, but with reduced T.

Typical usage:

- After generating a segment, cut 1 temporal slot from the end before feeding it as `prev_samples` to the next `WanImageToVideoSVIProFLF` node. This removes the problematic “hard‑locked + SVI” slot and makes segment stitching more stable.

This is a pragmatic workaround, not a theoretical limitation. If you discover a cleaner way to reconcile SVI 2 Pro motion with fully hard‑locked end slots, feel free to adapt or replace this node in your own workflows.

---

## Workflows

This repository includes two example workflows:

- One uses `KSampler (Advanced)`.
- The other uses `SamplerCustomAdvanced`.

They are intended to be equivalent in this context and are provided so you can pick whichever sampler node better fits your existing setup or personal preference.

---

## Installation

1. Go to your ComfyUI `custom_nodes` folder, for example:

   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/Well-Made/ComfyUI-Wan-SVI2Pro-FLF.git
   ```

3. Restart ComfyUI.

The nodes will appear under:

conditioning/video_models → Wan SVI 2 Pro FLF

latent/video → Wan Cut Last Slot

##License

This project is licensed under the GNU GPLv3.

Some logic is adapted from:

ComfyUI (Wan-related nodes, e.g. WanFirstLastFrameToVideo)

ComfyUI-KJNodes (WanImageToVideoSVIPro)

Both upstream projects are licensed under GPLv3, and this repository follows the same license.

## Related resources

- SVI tutorial (video):  
  [https://www.youtube.com/watch?v=-3DVJu72VhE](https://www.youtube.com/watch?v=-3DVJu72VhE)

- Stable Video Infinity main page:  
  [https://github.com/vita-epfl/Stable-Video-Infinity](https://github.com/vita-epfl/Stable-Video-Infinity)

- KJNodes for ComfyUI:  
  [https://github.com/kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

- Wan 2.2 SVI 2 Pro native workflow (JSON example by KJ):  
  [https://github.com/user-attachments/files/24359648/wan22_SVI_Pro_native_example_KJ.json](https://github.com/user-attachments/files/24359648/wan22_SVI_Pro_native_example_KJ.json)

- Wan fp8 I2V models for ComfyUI:  
  [https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/I2V](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/I2V)

- Lightx2v Wan 2.2 Distill LoRAs:  
  [https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/tree/main](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/tree/main)

- SVI 2.0 Pro LoRAs for Wan video:  
  [https://huggingface.co/Kijai/WanVideo_comfy/tree/main/LoRAs/Stable-Video-Infinity/v2.0](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/LoRAs/Stable-Video-Infinity/v2.0)

- Wan 2.2 I2V GGUF (for GGUF runtimes):  
  [https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/tree/main](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/tree/main)
