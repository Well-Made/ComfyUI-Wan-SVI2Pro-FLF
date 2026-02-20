import torch
import numpy as np

import comfy.model_management
import comfy.latent_formats

from nodes import MAX_RESOLUTION
from comfy_api.latest import io
import node_helpers


class WanImageToVideoSVIProFLF(io.ComfyNode):
    """
    WanImageToVideoSVIProFLF

    Combines Wan SVI 2 Pro–style motion continuity with FLF-style
    (First/Last Frame) control over the end of the clip.

    - Start: behaves like WanImageToVideoSVIPro (anchor frame + optional motion
      continuation from prev_samples, SVI Pro style).
    - End: behaves like WanFirstLastFrameToVideoLatent / FLF2V, where the last
      temporal slots are hard-locked to the provided end_samples block.

    Typical use:
    - anchor_samples: first frame (or short latent clip) of this segment.
    - prev_samples: tail latents from the previous segment to continue motion.
    - end_samples: last frame (or short latent clip) that should define the end.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanImageToVideoSVIProFLF",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input(
                    "length",
                    default=81,
                    min=1,
                    max=MAX_RESOLUTION,
                    step=4,
                    tooltip=(
                        "Target video length in frames. "
                        "Wan video latents use a temporal stride of 4."
                    ),
                ),
                io.Latent.Input(
                    "prev_samples",
                    tooltip=(
                        "Previous segment latents (B,C,T,H,W). "
                        "The last motion_latent_count temporal slots will be "
                        "used to continue motion (SVI Pro style)."
                    ),
                ),
                io.Latent.Input(
                    "anchor_samples",
                    tooltip=(
                        "Anchor latent(s) for the current segment (B,C,T,H,W). "
                        "Typically the first frame (or a very short clip) of this segment."
                    ),
                ),
                io.Latent.Input(
                    "end_samples",
                    optional=True,
                    tooltip=(
                        "Optional target end latent(s), in Wan video latent format (B,C,T,H,W). "
                        "The last temporal slots of this tensor will define the last slots "
                        "of the clip (FLF-style hard lock)."
                    ),
                ),
                io.Int.Input(
                    "motion_latent_count",
                    default=1,
                    min=0,
                    max=16,
                    step=1,
                    tooltip=(
                        "How many last temporal latent slots to copy from prev_samples. "
                        "0 = no motion continuity; 1–2 is typical for smooth transitions."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        length,
        prev_samples=None,
        anchor_samples=None,
        motion_latent_count=1,
        end_samples=None,
    ) -> io.NodeOutput:
        """
        Execution logic:

        - Start: build a base temporal block from anchor_samples plus an optional
          motion tail taken from the end of prev_samples (SVI Pro continuity).
        - Padding: extend this block to the required temporal length for Wan
          (total_latents) using Wan-formatted zero latents.
        - End: overwrite the last temporal slots with end_samples (if provided),
          and hard-lock them via the concat_mask (FLF-style control).
        """

        # Anchor latent for this segment: [B, C, T_anchor, H, W]
        anchor_latent = anchor_samples["samples"].clone()
        B, C, T_anchor, H, W = anchor_latent.shape

        # Compute number of temporal latent slots given Wan's stride=4.
        # Example: length=41 frames -> total_latents=11.
        total_latents = (length - 1) // 4 + 1
        device = anchor_latent.device
        dtype = anchor_latent.dtype

        # ---------------------------------------------------------------------
        # 1) Base temporal block:
        #    anchor + optional motion tail from prev_samples (SVI Pro logic).
        # ---------------------------------------------------------------------
        if prev_samples is None or motion_latent_count == 0:
            # No previous segment or motion continuity disabled:
            # start purely from the anchor latent block.
            base_latent = anchor_latent
        else:
            # Previous segment latents: [B, C, T_prev, H, W]
            prev_latent = prev_samples["samples"].clone()
            T_prev = prev_latent.shape[2]

            # We can only take as many time slots as prev_latent actually has.
            motion_count = min(motion_latent_count, T_prev)

            if motion_count > 0:
                # Take the last `motion_count` temporal slots as motion tail:
                # shape [B, C, T_m, H, W].
                motion_latent = prev_latent[:, :, -motion_count:]
                # Concatenate anchor and motion along the temporal axis.
                base_latent = torch.cat([anchor_latent, motion_latent], dim=2)
            else:
                # motion_latent_count requested 0 or prev had T_prev=0 (edge case)
                base_latent = anchor_latent

        T_base = base_latent.shape[2]
        padding_size = max(total_latents - T_base, 0)

        # ---------------------------------------------------------------------
        # 2) Pad with Wan-format zeros to reach total_latents.
        #    This ensures the conditioning latent has the exact temporal
        #    length expected by the Wan sampler.
        # ---------------------------------------------------------------------
        if padding_size > 0:
            # Create zero latents and convert them to Wan's internal format.
            padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
            padding = comfy.latent_formats.Wan21().process_out(padding)
            # Append padding after the base sequence.
            image_cond_latent = torch.cat([base_latent, padding], dim=2)
        else:
            # If base_latent is longer than needed, truncate it.
            image_cond_latent = base_latent[:, :, :total_latents]

        # Safety clamp in case of any mismatch.
        if image_cond_latent.shape[2] != total_latents:
            image_cond_latent = image_cond_latent[:, :, :total_latents]

        # ---------------------------------------------------------------------
        # 3) End behavior like WanFirstLastFrameToVideoLatent (FLF-style):
        #    hard-lock the last frames to end_samples, if provided.
        # ---------------------------------------------------------------------
        end_t_fix = 0
        if end_samples is not None:
            # [B_end, C, T_end, H, W] or [1, C, T_end, H, W]
            end_latent = end_samples["samples"].clone()

            # If end_latent has batch size 1 but our segment batch is larger,
            # repeat it across the batch dimension.
            if end_latent.shape[0] == 1 and B > 1:
                end_latent = end_latent.repeat(B, 1, 1, 1, 1)

            # Check that channels and spatial dimensions match.
            if (
                end_latent.shape[1] == C
                and end_latent.shape[3] == H
                and end_latent.shape[4] == W
            ):
                T_end = end_latent.shape[2]

                # FLF2V-style: fix as many frames as the end block provides,
                # but not more than total_latents.
                end_t_fix = min(T_end, total_latents)

                if end_t_fix > 0:
                    # Overwrite the last `end_t_fix` temporal slots of
                    # image_cond_latent with the last slots of end_latent.
                    image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]
            else:
                # Shape mismatch: skip end locking for safety.
                end_t_fix = 0

        # ---------------------------------------------------------------------
        # 4) Empty latent to be filled by the Wan sampler.
        #    Shape: [B, 16, total_latents, H, W] for Wan video UNet.
        # ---------------------------------------------------------------------
        empty_latent = torch.zeros(
            [B, 16, total_latents, H, W],
            device=comfy.model_management.intermediate_device(),
        )

        # ---------------------------------------------------------------------
        # 5) Mask:
        #    - First temporal slot (anchor) is fixed: mask=0 at t=0.
        #    - Last end_t_fix slots are fixed if end_samples is provided.
        #
        # Wan / SVI Pro use concat_mask==0 to mark slots that should stay
        # equal to concat_latent_image during sampling.
        # ---------------------------------------------------------------------
        mask = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)

        # Lock the first temporal slot (anchor).
        mask[:, :, :1] = 0.0

        # Lock the last temporal slots corresponding to end_samples.
        if end_t_fix > 0:
            mask[:, :, -end_t_fix:] = 0.0

        # ---------------------------------------------------------------------
        # 6) Inject concat_latent_image and concat_mask into conditioning.
        #    Both positive and negative conditioning receive the same concat
        #    latent and mask so the sampler can enforce structure consistently.
        # ---------------------------------------------------------------------
        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask,
            },
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask,
            },
        )

        out_latent = {"samples": empty_latent}
        return io.NodeOutput(positive, negative, out_latent)


class WanCutLastSlot(io.ComfyNode):
    """
    WanCutLastSlot

    Utility node for Wan video latents: trims a number of temporal slots
    from the end of a latent clip.

    In Wan 2.2, 1 temporal slot corresponds to 4 video frames (stride = 4),
    so cutting 1 slot effectively removes the last 4 frames worth of latent
    information from the sequence.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanCutLastSlot",
            category="latent/video",
            inputs=[
                io.Latent.Input(
                    "latents",
                    tooltip=(
                        "Wan video latents to trim (dict with 'samples' key, "
                        "shape [B,C,T,H,W]). The last temporal slots will be removed."
                    ),
                ),
                io.Int.Input(
                    "slots_to_cut",
                    default=1,
                    min=1,
                    max=64,
                    step=1,
                    tooltip=(
                        "How many last temporal slots (T dimension) to cut. "
                        "In Wan 2.2, 1 slot ≈ 4 frames."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="trimmed_latents"),
            ],
        )

    @classmethod
    def execute(cls, latents, slots_to_cut=1) -> io.NodeOutput:
        """
        Remove `slots_to_cut` temporal slots from the end of the latent clip.

        - Input:  latents["samples"] with shape [B, C, T, H, W].
        - Output: same dict, but with T' = T - n, where
          n = min(slots_to_cut, T - 1), so at least 1 slot always remains.
        """

        x = latents["samples"]
        B, C, T, H, W = x.shape

        # Do not allow cutting all temporal slots:
        # ensure at least 1 slot remains.
        n = max(1, min(slots_to_cut, T - 1))

        # Keep everything except the last `n` slots.
        x_trim = x[:, :, : T - n].clone()

        out = latents.copy()
        out["samples"] = x_trim
        return io.NodeOutput(out)
