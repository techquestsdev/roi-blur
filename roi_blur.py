#!/usr/bin/env python3
"""
ROI Blur Tool
=============

A simple OpenCV-based utility for interactively selecting regions of interest (ROIs)
in an image and applying Gaussian blur to those regions. Useful for privacy redaction,
hiding sensitive information, or artistic effects.

Usage:
    python roi_blur.py input.jpg output.jpg
    python roi_blur.py photo.png blurred.png --ksize 51 --sigma 50

Controls:
    - Click and drag to draw ROI rectangle
    - ENTER/SPACE: Confirm selection
    - ESC: Cancel current selection
    - 'u': Undo last ROI
    - 'q': Finish selection and apply blur

Author: Andre Nogueira
License: GPL-3.0
Repository: https://github.com/techquestsdev/roi-blur
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

# Supported image formats for output
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Default blur parameters
DEFAULT_KERNEL_SIZE: int = 23
DEFAULT_SIGMA: float = 30.0

# UI Colors (BGR format)
COLOR_ROI_CONFIRMED: tuple[int, int, int] = (0, 255, 0)  # Green
COLOR_ROI_THICKNESS: int = 2

# Window names
WINDOW_SELECT: str = "ROI Selection - Press 'q' when done, 'u' to undo"
WINDOW_RESULT: str = "Blurred Result - Press any key to exit"


# =============================================================================
# Core Functions
# =============================================================================


def validate_kernel_size(ksize: int) -> int:
    """
    Ensure kernel size is positive and odd (required by OpenCV GaussianBlur).

    Args:
        ksize: The desired kernel size.

    Returns:
        A valid odd kernel size >= 1.

    Examples:
        >>> validate_kernel_size(23)
        23
        >>> validate_kernel_size(24)
        25
        >>> validate_kernel_size(0)
        1
    """
    # Ensure minimum value of 1
    ksize = max(1, ksize)

    # Make odd if even (OpenCV requirement)
    if ksize % 2 == 0:
        ksize += 1

    return ksize


def blur_boxes(
    image: NDArray[np.uint8],
    boxes: list[tuple[int, int, int, int]],
    ksize: int = DEFAULT_KERNEL_SIZE,
    sigma: float = DEFAULT_SIGMA,
) -> NDArray[np.uint8]:
    """
    Apply Gaussian blur to specified rectangular regions in an image.

    This function creates a copy of the input image and applies Gaussian blur
    independently to each specified region. Regions are clamped to image bounds
    to prevent out-of-bounds access.

    Args:
        image: Input image as a NumPy array (BGR format, as read by cv2.imread).
        boxes: List of ROI tuples, each containing (x, y, width, height) where
               (x, y) is the top-left corner coordinate.
        ksize: Gaussian kernel size (must be odd; will be auto-corrected if even).
               Larger values = more blur. Default is 23.
        sigma: Gaussian kernel standard deviation in both X and Y directions.
               Larger values = more blur spread. Default is 30.0.

    Returns:
        A new image array with blur applied to the specified regions.
        The original image is not modified.

    Note:
        - Empty or invalid boxes (width <= 0 or height <= 0) are silently skipped.
        - Boxes extending beyond image boundaries are automatically clamped.
        - Uses BORDER_REPLICATE to avoid edge artifacts in blurred regions.

    Example:
        >>> img = cv2.imread("photo.jpg")
        >>> boxes = [(100, 100, 200, 150), (400, 300, 100, 100)]
        >>> result = blur_boxes(img, boxes, ksize=31, sigma=40)
        >>> cv2.imwrite("blurred.jpg", result)
    """
    # Create output copy to avoid modifying original
    out: NDArray[np.uint8] = image.copy()
    h_img, w_img = out.shape[:2]

    # Validate and prepare kernel size (must be odd for GaussianBlur)
    ksize = validate_kernel_size(ksize)
    kernel: tuple[int, int] = (ksize, ksize)

    for x, y, w, h in boxes:
        # Convert to integers (safety measure for float inputs)
        x, y, w, h = map(int, (x, y, w, h))

        # Skip invalid boxes with non-positive dimensions
        if w <= 0 or h <= 0:
            continue

        # Clamp ROI coordinates to image bounds
        # This prevents index errors when ROI extends beyond image edges
        x1: int = max(0, min(x, w_img))
        y1: int = max(0, min(y, h_img))
        x2: int = max(0, min(x + w, w_img))
        y2: int = max(0, min(y + h, h_img))

        # Skip if clamping resulted in zero-area region
        if x2 <= x1 or y2 <= y1:
            continue

        # Extract ROI (copy to ensure blur operation doesn't leak to adjacent pixels)
        roi: NDArray[np.uint8] = out[y1:y2, x1:x2].copy()

        # Skip empty ROIs (additional safety check)
        if roi.size == 0:
            continue

        # Apply Gaussian blur to the ROI
        # BORDER_REPLICATE extends edge pixels to avoid dark borders
        roi_blur: NDArray[np.uint8] = cv2.GaussianBlur(
            roi,
            kernel,
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

        # Place blurred ROI back into output image
        out[y1:y2, x1:x2] = roi_blur

    return out


def interactive_roi_selection(image: NDArray[np.uint8]) -> list[tuple[int, int, int, int]]:
    """
    Interactively select multiple ROIs on an image using OpenCV's selectROI.

    Displays the image in a window and allows the user to draw rectangles
    to define regions of interest. Selected regions are highlighted with
    green rectangles.

    Args:
        image: Input image to select ROIs from (BGR format).

    Returns:
        List of ROI tuples, each containing (x, y, width, height).

    Controls:
        - Click and drag: Draw selection rectangle
        - ENTER or SPACE: Confirm current selection
        - ESC: Cancel current selection
        - 'u': Undo last confirmed selection
        - 'q': Finish and return all selections
    """
    # Working copy for drawing confirmed ROIs
    display_image: NDArray[np.uint8] = image.copy()
    rois: list[tuple[int, int, int, int]] = []

    # Store display states for undo functionality
    display_history: list[NDArray[np.uint8]] = [display_image.copy()]

    print("\n" + "=" * 60)
    print("ROI SELECTION MODE")
    print("=" * 60)
    print("Controls:")
    print("  - Click and drag to draw ROI rectangle")
    print("  - ENTER/SPACE : Confirm selection")
    print("  - ESC         : Cancel current selection")
    print("  - 'u'         : Undo last ROI")
    print("  - 'q'         : Finish and apply blur")
    print("=" * 60 + "\n")

    while True:
        # Show current state and wait briefly for window to update
        cv2.imshow(WINDOW_SELECT, display_image)
        cv2.waitKey(1)

        # Launch interactive ROI selection
        # Returns (x, y, w, h) or (0, 0, 0, 0) if cancelled
        box: tuple[int, int, int, int] = cv2.selectROI(
            WINDOW_SELECT,
            display_image,
            fromCenter=False,
            showCrosshair=True,
        )
        x, y, w, h = map(int, box)

        # Check if selection was cancelled or empty
        if w == 0 or h == 0:
            print("[INFO] Empty or cancelled selection - ignored")
        else:
            # Save the ROI
            rois.append((x, y, w, h))

            # Draw confirmed ROI rectangle on display image
            cv2.rectangle(
                display_image,
                (x, y),
                (x + w, y + h),
                COLOR_ROI_CONFIRMED,
                COLOR_ROI_THICKNESS,
            )

            # Save state for undo
            display_history.append(display_image.copy())

            print(f"[OK] ROI #{len(rois)} saved: x={x}, y={y}, w={w}, h={h}")

        # Update display and wait for user command
        cv2.imshow(WINDOW_SELECT, display_image)
        key: int = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            # Finish selection
            print(f"\n[DONE] Selection complete. Total ROIs: {len(rois)}")
            break
        elif key == ord("u"):
            # Undo last ROI
            if rois:
                removed = rois.pop()
                display_history.pop()
                display_image = display_history[-1].copy()
                print(f"[UNDO] Removed ROI: x={removed[0]}, y={removed[1]}, "
                      f"w={removed[2]}, h={removed[3]}")
            else:
                print("[UNDO] Nothing to undo")

    # Clean up selection window
    cv2.destroyAllWindows()

    return rois


def validate_output_path(output_path: str) -> Path:
    """
    Validate the output file path.

    Checks that:
    - The parent directory exists
    - The file extension is a supported image format

    Args:
        output_path: Path string for the output file.

    Returns:
        Validated Path object.

    Raises:
        SystemExit: If directory doesn't exist or extension is unsupported.
    """
    path = Path(output_path)

    # Check parent directory exists
    if not path.parent.exists():
        raise SystemExit(f"[ERROR] Output directory does not exist: {path.parent}")

    # Validate file extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise SystemExit(
            f"[ERROR] Unsupported output format: '{ext}'\n"
            f"        Supported formats: {supported}"
        )

    return path


# Global variable to store image metadata (ICC profile, etc.) for preservation
_image_metadata: dict = {}


def load_image(image_path: str) -> NDArray[np.uint8]:
    """
    Load an image from disk, preserving original colors.

    Uses Pillow to load the image to properly handle ICC color profiles
    and color modes, then converts to OpenCV BGR format.

    Args:
        image_path: Path to the input image file.

    Returns:
        Image as a NumPy array in BGR format (or BGRA if alpha channel present).

    Raises:
        SystemExit: If file doesn't exist or cannot be read.
    """
    global _image_metadata
    path = Path(image_path)

    if not path.exists():
        raise SystemExit(f"[ERROR] Input file not found: {image_path}")

    try:
        # Use Pillow to load image - it handles ICC profiles correctly
        pil_image = Image.open(str(path))

        # Store metadata for preservation when saving
        _image_metadata = {
            'icc_profile': pil_image.info.get('icc_profile'),
            'exif': pil_image.info.get('exif'),
            'dpi': pil_image.info.get('dpi'),
        }

        # Convert palette/indexed images to RGB
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')

        # Handle different color modes
        if pil_image.mode == 'L':  # Grayscale
            # Convert to RGB then to BGR
            pil_image = pil_image.convert('RGB')
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif pil_image.mode == 'LA':  # Grayscale with alpha
            pil_image = pil_image.convert('RGBA')
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif pil_image.mode == 'RGB':
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif pil_image.mode == 'RGBA':
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif pil_image.mode == 'CMYK':
            pil_image = pil_image.convert('RGB')
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # For any other mode, convert to RGB
            pil_image = pil_image.convert('RGB')
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to read image: {image_path}\n        {e}")


def save_image(image: NDArray[np.uint8], output_path: Path) -> None:
    """
    Save an image to disk with maximum color fidelity.

    Preserves ICC color profile and other metadata from the original image.

    Args:
        image: Image array to save (BGR format).
        output_path: Path object for the output file.

    Raises:
        SystemExit: If the image cannot be saved.
    """
    global _image_metadata

    # Convert from OpenCV BGR/BGRA format back to RGB/RGBA for saving
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # BGRA to RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            pil_image = Image.fromarray(image_rgb, mode='RGBA')
        else:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb, mode='RGB')
    else:
        # Grayscale
        pil_image = Image.fromarray(image, mode='L')

    # Determine appropriate save parameters based on file extension
    ext = output_path.suffix.lower()
    save_kwargs: dict = {}

    if ext in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 100, "subsampling": 0}  # 4:4:4 no subsampling
    elif ext == ".png":
        save_kwargs = {"compress_level": 3}
    elif ext == ".webp":
        save_kwargs = {"lossless": True}

    # Preserve ICC color profile if present
    if _image_metadata.get('icc_profile'):
        save_kwargs['icc_profile'] = _image_metadata['icc_profile']

    # Preserve EXIF data if present (for JPEG/TIFF)
    if _image_metadata.get('exif') and ext in {".jpg", ".jpeg", ".tiff", ".tif"}:
        save_kwargs['exif'] = _image_metadata['exif']

    try:
        pil_image.save(str(output_path), **save_kwargs)
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to save image: {output_path}\n        {e}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: List of argument strings. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace.

    Raises:
        SystemExit: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        prog="roi_blur",
        description="Interactively select regions in an image and apply Gaussian blur.",
        epilog="Example: %(prog)s input.jpg output.jpg --ksize 31 --sigma 40",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=str,
        metavar="INPUT",
        help="Path to the input image file",
    )

    parser.add_argument(
        "output",
        type=str,
        metavar="OUTPUT",
        help="Path for the output image file",
    )

    parser.add_argument(
        "-k", "--ksize",
        type=int,
        default=DEFAULT_KERNEL_SIZE,
        metavar="N",
        help=f"Blur kernel size (positive odd integer, default: {DEFAULT_KERNEL_SIZE})",
    )

    parser.add_argument(
        "-s", "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        metavar="N",
        help=f"Blur sigma/strength (positive float, default: {DEFAULT_SIGMA})",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    args = parser.parse_args(argv)

    # Validate kernel size
    if args.ksize <= 0:
        parser.error(f"Kernel size must be positive, got: {args.ksize}")

    # Validate sigma
    if args.sigma < 0:
        parser.error(f"Sigma must be non-negative, got: {args.sigma}")

    return args


def main(argv: list[str] | None = None) -> None:
    """
    Main application entry point.

    Orchestrates the workflow:
    1. Parse command-line arguments
    2. Load input image
    3. Interactive ROI selection
    4. Apply blur to selected regions
    5. Save result

    Args:
        argv: Optional list of command-line arguments.
              If None, uses sys.argv[1:].
    """
    # Parse command-line arguments
    args = parse_args(argv)

    # Validate output path first (fail fast)
    output_path = validate_output_path(args.output)

    # Load input image
    print(f"[INFO] Loading image: {args.input}")
    image: NDArray[np.uint8] = load_image(args.input)
    print(f"[INFO] Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # Interactive ROI selection
    rois = interactive_roi_selection(image)

    # Check if any ROIs were selected
    if not rois:
        print("[WARNING] No ROIs selected. Output will be identical to input.")

    # Apply blur to selected regions
    print(f"[INFO] Applying blur (kernel={args.ksize}, sigma={args.sigma})...")
    result: NDArray[np.uint8] = blur_boxes(
        image,
        rois,
        ksize=args.ksize,
        sigma=args.sigma,
    )

    # Save output
    save_image(result, output_path)
    print(f"[OK] Saved: {output_path}")

    # Display result
    print("[INFO] Displaying result. Press any key to exit.")
    cv2.imshow(WINDOW_RESULT, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
