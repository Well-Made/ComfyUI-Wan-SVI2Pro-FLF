from .nodes import WanImageToVideoSVIProFLF, WanCutLastSlot


# Mapping from internal node IDs to Python classes
NODE_CLASS_MAPPINGS = {
    "WanImageToVideoSVIProFLF": WanImageToVideoSVIProFLF,
    "WanCutLastSlot": WanCutLastSlot,
}

# Human‑readable names shown in the ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanImageToVideoSVIProFLF": "Wan SVI 2 Pro FLF",
    "WanCutLastSlot": "Wan Cut Last Slot",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
