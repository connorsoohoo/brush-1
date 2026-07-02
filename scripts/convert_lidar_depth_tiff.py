#!/usr/bin/env python3
"""Convert Splat King raw LiDAR binary depth files into float32 TIFF depth maps.

For each capture in the sensor directory, reads the raw LiDAR depth buffer (a
float32 array) and, optionally, its confidence buffer (uint8), masks out
low-confidence samples, and bilinearly upscales the depth to the capture's full
image resolution. Buffer dimensions and the output resolution are all read from
the capture's sidecar JSON (auxiliaryOutputs and metadata.camera.imageResolution),
so no source images are needed. The result is written as a single-channel
float32 (.tiff) depth map that the Brush depth-loss pipeline can consume.

Inputs (per capture, matched by filename stem <name> in <sensor_dir>):
    <name>.json                 sidecar metadata; gives buffer dims + output W x H (required)
    <name>_depth.bin            float32 depth buffer, row-major (required)
    <name>_confidence.bin       uint8 confidence buffer, values 0/1/2 (optional)

Output:
    <output_dir>/<name>.tiff    W x H float32 depth, meters; masked pixels are 0.0.
                                Captures missing a sidecar JSON are skipped.

Usage:
    # Convert all captures, keeping confidence >= 1 (the default):
    python scripts/convert_lidar_depth_tiff.py \
        --sensor_dir /path/to/sensor_data \
        --output_dir /path/to/COLMAP_Text_Model/depth_lidar

    # Keep only the highest-confidence LiDAR samples:
    python scripts/convert_lidar_depth_tiff.py --min_confidence 2 ...
"""

import os
import glob
import json
import numpy as np
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Splat King raw LiDAR binary depth files to float32 TIFFs.")
    parser.add_argument(
        "--sensor_dir",
        type=str,
        required=True,
        help="Directory containing raw <name>.json, <name>_depth.bin, and <name>_confidence.bin files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted tiff files."
    )
    parser.add_argument(
        "--min_confidence",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Minimum confidence level to keep (0, 1, or 2). Low confidence depths will be masked to 0.0."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    depth_bins = sorted(glob.glob(os.path.join(args.sensor_dir, "*_depth.bin")))

    print(f"Found {len(depth_bins)} depth captures to process.")
    converted_count = 0

    for depth_bin_path in depth_bins:
        base_name = os.path.basename(depth_bin_path)[: -len("_depth.bin")]
        json_path = os.path.join(args.sensor_dir, f"{base_name}.json")
        conf_bin_path = os.path.join(args.sensor_dir, f"{base_name}_confidence.bin")

        if not os.path.exists(json_path):
            continue

        # Read the output resolution and buffer dimensions from the sidecar metadata.
        with open(json_path) as f:
            meta = json.load(f)
        resolution = meta["metadata"]["camera"]["imageResolution"]
        w, h = resolution["width"], resolution["height"]
        buffers = {aux["type"]: aux for aux in meta["auxiliaryOutputs"]}

        # Load raw depth float32 at the buffer's declared dimensions
        depth_buf = buffers["depth"]
        depth_data = np.fromfile(depth_bin_path, dtype=np.float32).reshape(
            depth_buf["height"], depth_buf["width"]
        )

        # Load raw confidence uint8 and mask out low confidence values
        if os.path.exists(conf_bin_path):
            conf_buf = buffers["depthConfidence"]
            conf_data = np.fromfile(conf_bin_path, dtype=np.uint8).reshape(
                conf_buf["height"], conf_buf["width"]
            )
            depth_data[conf_data < args.min_confidence] = 0.0

        # Convert to PIL Image in mode 'F' (float32) and upscale to match original resolution
        depth_pil = Image.fromarray(depth_data, mode='F')
        depth_resized = depth_pil.resize((w, h), resample=Image.Resampling.BILINEAR)

        # Save to output path as float32 TIFF
        output_path = os.path.join(args.output_dir, f"{base_name}.tiff")
        depth_resized.save(output_path)
        converted_count += 1

    print(f"Done! Converted {converted_count} LiDAR depths and saved them to {args.output_dir}")

if __name__ == "__main__":
    main()
