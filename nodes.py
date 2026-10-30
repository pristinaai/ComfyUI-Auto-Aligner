import numpy as np
import torch

from .align import align_pair

HARMONIZE = {
    "Off": 0.0,
    "Low": 0.35,
    "Med": 0.65,
    "High": 0.9,
    "Full": 1.0,
}


class AutoAligner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "destination": ("IMAGE",),
                "resize": ("BOOLEAN", {"default": True, "tooltip": "Scale the source to fit the destination."}),
                "position": ("BOOLEAN", {"default": True, "tooltip": "Shift the source onto the destination."}),
                "rotate": ("BOOLEAN", {"default": True, "tooltip": "Rotate the source to match the destination."}),
                "puppet_warp": ("BOOLEAN", {"default": False, "tooltip": "Bend the source locally after the main fit."}),
                "warp_grid": ("INT", {
                    "default": 12,
                    "min": 3,
                    "max": 24,
                    "step": 1,
                    "tooltip": "Puppet warp grid. 3 is smooth, 24 follows small mismatches.",
                }),
                "colour_match": (list(HARMONIZE.keys()), {
                    "default": "Off",
                    "tooltip": "Match the source lighting and colour to the destination.",
                }),
                "auto_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep only the parts of the source that differ from the destination.",
                }),
                "mask_sensitivity": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Higher keeps smaller differences in the auto mask.",
                }),
                "features": (["High", "Fast", "Rigorous"], {
                    "default": "High",
                    "tooltip": "High uses up to 16000 features at 3200px. Rigorous uses 32000 at 5120px, then fits the warp to the pixels. Fast uses 4000 at 1600px.",
                }),
                "lock_aspect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use one scale for width and height. Off lets them scale separately.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("aligned", "mask")
    FUNCTION = "align"
    CATEGORY = "image/align"
    DESCRIPTION = "Align the source image onto the destination. Output matches the destination size."

    def align(
        self,
        source,
        destination,
        resize,
        position,
        rotate,
        puppet_warp,
        warp_grid,
        colour_match,
        auto_mask,
        mask_sensitivity,
        features,
        lock_aspect,
    ):
        src = _as_batch(source)
        dst = _as_batch(destination)
        count = max(src.shape[0], dst.shape[0])
        if src.shape[0] not in (1, count) or dst.shape[0] not in (1, count):
            raise ValueError(
                f"Batch sizes must match or one side must be 1 (source {src.shape[0]}, destination {dst.shape[0]})."
            )

        options = {
            "resize": bool(resize),
            "lock_aspect": bool(lock_aspect),
            "position": bool(position),
            "rotate": bool(rotate),
            "puppet_warp": bool(puppet_warp),
            "warp_grid": int(warp_grid),
            "harmonize": HARMONIZE.get(colour_match, 0.0),
            "auto_mask": bool(auto_mask),
            "mask_sensitivity": int(mask_sensitivity) / 100.0,
            "features": {"High": "high", "Fast": "fast", "Rigorous": "rigorous"}[features],
        }

        images = []
        masks = []
        for i in range(count):
            dest = _frame_u8(dst, i)
            try:
                rgb, alpha = align_pair(_frame_u8(src, i), dest, options)
            except RuntimeError as err:
                if "Too few inliers" not in str(err):
                    raise
                print(f"Auto Aligner: {err} Passing the destination through with an empty mask.")
                rgb = dest.astype(np.float32) / 255.0
                alpha = np.zeros(dest.shape[:2], np.float32)
            images.append(rgb)
            masks.append(alpha)

        image = torch.from_numpy(np.stack(images).astype(np.float32))
        mask = torch.from_numpy(np.stack(masks).astype(np.float32))
        return (image, mask)


def _as_batch(image):
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    return np.clip(arr, 0.0, 1.0)


def _frame_u8(batch, index):
    frame = batch[index % batch.shape[0], ..., :3]
    return np.clip(np.round(frame * 255.0), 0, 255).astype(np.uint8)


NODE_CLASS_MAPPINGS = {"AutoAligner": AutoAligner}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoAligner": "Auto Aligner"}
