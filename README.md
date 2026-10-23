# Auto Aligner

ComfyUI node that aligns a source image onto a destination image. The result is the destination resolution. Empty areas stay in the mask.

## Install

In ComfyUI Manager, choose **Install via Git URL** and paste:

`https://github.com/pristinaai/ComfyUI-Auto-Aligner`

Restart ComfyUI. It installs `numpy` and `opencv-python` from `requirements.txt`.

After this node is on the Comfy Registry, search for **Auto Aligner** in Manager, or run `comfy node install auto-aligner`.

To update, use Manager’s update button.

The node is **Auto Aligner** under `image/align`.

Inputs: resize, position, rotate, puppet warp, warp grid, colour match, auto mask, mask sensitivity, and features. Features defaults to High: up to 16000 keypoints at 3200px. Fast is 4000 keypoints at 1600px.

Outputs: `aligned` (IMAGE) and `mask` (MASK). Composite `aligned` over the destination with the mask.

If no alignment can be found, the node passes the destination through and returns an empty mask.
