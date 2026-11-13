# Reference: stripped from ~/Projects/aruco-grasp-annotator/src/aruco_annotator/utils/aruco_utils.py
"""Generate ArUco marker PNG images on demand. Called by extension at runtime."""

import cv2
import numpy as np
from pathlib import Path


ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}


def generate_marker_png(
    dictionary: str,
    marker_id: int,
    output_path: str,
    size_pixels: int = 200,
    border_bits: int = 1,
) -> str:
    """Generate an ArUco marker PNG if it doesn't already exist.

    Args:
        dictionary: ArUco dictionary name (e.g. "DICT_4X4_50").
        marker_id: Marker ID (0-based).
        output_path: Full path for the output PNG.
        size_pixels: Pixel size of the generated image.
        border_bits: Width of the white border in marker bits.

    Returns:
        The output_path (for convenience).
    """
    out = Path(output_path)
    if out.exists():
        return str(out)

    out.parent.mkdir(parents=True, exist_ok=True)

    if dictionary not in ARUCO_DICTS:
        raise ValueError(f"Unknown dictionary: {dictionary}. Available: {list(ARUCO_DICTS.keys())}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels, borderBits=border_bits)

    # White quiet zone around the marker. Set to 0 to print the marker
    # edge-to-edge with no white margin; cv2.copyMakeBorder is a no-op
    # at width 0 but kept here so the white-zone amount is one constant
    # to flip if a quiet zone is needed again for ArUco detection
    # robustness.
    border_px = 0
    bordered = cv2.copyMakeBorder(
        marker_img, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=255,
    )

    cv2.imwrite(str(out), bordered)
    return str(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ArUco marker PNGs")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("--id", type=int, required=True, help="Marker ID")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--size", type=int, default=200, help="Image size in pixels")
    args = parser.parse_args()

    path = generate_marker_png(args.dict, args.id, args.output, args.size)
    print(f"Generated: {path}")
