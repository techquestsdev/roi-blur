#!/usr/bin/env python3
"""
Tests for ROI Blur Tool
=======================

Run with: pytest test_roi_blur.py -v
Run with coverage: pytest test_roi_blur.py --cov=roi_blur --cov-report=term-missing
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# Import the module under test
import roi_blur
from roi_blur import (
    DEFAULT_KERNEL_SIZE,
    DEFAULT_SIGMA,
    SUPPORTED_EXTENSIONS,
    blur_boxes,
    validate_kernel_size,
    validate_output_path,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a simple test image (100x100 RGB with distinct colored regions)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Red region (top-left quadrant)
    img[0:50, 0:50] = [0, 0, 255]
    # Green region (top-right quadrant)
    img[0:50, 50:100] = [0, 255, 0]
    # Blue region (bottom-left quadrant)
    img[50:100, 0:50] = [255, 0, 0]
    # White region (bottom-right quadrant)
    img[50:100, 50:100] = [255, 255, 255]
    return img


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Create a grayscale test image with sharp edges for blur visibility."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Create a checkerboard pattern so blur is visible
    img[0:50, 0:50] = 0  # Top-left: black
    img[0:50, 50:100] = 255  # Top-right: white
    img[50:100, 0:50] = 255  # Bottom-left: white
    img[50:100, 50:100] = 0  # Bottom-right: black
    return img


@pytest.fixture
def temp_image_file(tmp_path: Path, sample_image: np.ndarray) -> Path:
    """Create a temporary image file for testing."""
    filepath = tmp_path / "test_input.png"
    cv2.imwrite(str(filepath), sample_image)
    return filepath


@pytest.fixture
def temp_output_path(tmp_path: Path) -> Path:
    """Create a temporary output path for testing."""
    return tmp_path / "test_output.png"


# =============================================================================
# Tests: validate_kernel_size
# =============================================================================


class TestValidateKernelSize:
    """Tests for the validate_kernel_size function."""

    def test_odd_number_unchanged(self):
        """Odd numbers should remain unchanged."""
        assert validate_kernel_size(1) == 1
        assert validate_kernel_size(3) == 3
        assert validate_kernel_size(23) == 23
        assert validate_kernel_size(101) == 101

    def test_even_number_incremented(self):
        """Even numbers should be incremented to next odd."""
        assert validate_kernel_size(2) == 3
        assert validate_kernel_size(24) == 25
        assert validate_kernel_size(100) == 101

    def test_zero_becomes_one(self):
        """Zero should become 1."""
        assert validate_kernel_size(0) == 1

    def test_negative_becomes_one(self):
        """Negative numbers should become 1."""
        assert validate_kernel_size(-1) == 1
        assert validate_kernel_size(-100) == 1

    def test_large_values(self):
        """Large values should work correctly."""
        assert validate_kernel_size(999) == 999
        assert validate_kernel_size(1000) == 1001


# =============================================================================
# Tests: blur_boxes
# =============================================================================


class TestBlurBoxes:
    """Tests for the blur_boxes function."""

    def test_no_boxes_returns_copy(self, sample_image: np.ndarray):
        """With no boxes, should return an identical copy."""
        result = blur_boxes(sample_image, [])
        np.testing.assert_array_equal(result, sample_image)
        # Verify it's a copy, not the same object
        assert result is not sample_image

    def test_original_not_modified(self, sample_image: np.ndarray):
        """Original image should not be modified."""
        original_copy = sample_image.copy()
        boxes = [(10, 10, 30, 30)]
        blur_boxes(sample_image, boxes)
        np.testing.assert_array_equal(sample_image, original_copy)

    def test_single_box_is_blurred(self, sample_image: np.ndarray):
        """A single box region should be blurred (different from original)."""
        # Use a box that crosses color boundaries (quadrant edges at 50,50)
        boxes = [(40, 40, 20, 20)]  # Crosses all 4 quadrants
        result = blur_boxes(sample_image, boxes, ksize=11, sigma=10)

        # The blurred region should differ from original
        original_roi = sample_image[40:60, 40:60]
        result_roi = result[40:60, 40:60]
        assert not np.array_equal(original_roi, result_roi)

        # Region outside the box should be unchanged
        np.testing.assert_array_equal(sample_image[70:90, 70:90], result[70:90, 70:90])

    def test_multiple_boxes(self, sample_image: np.ndarray):
        """Multiple boxes should all be blurred."""
        # Use boxes that cross quadrant boundaries to ensure blur is visible
        boxes = [(40, 10, 20, 30), (10, 40, 30, 20)]  # Cross horizontal and vertical edges
        result = blur_boxes(sample_image, boxes, ksize=5, sigma=5)

        # Both regions should be different from original (they cross color boundaries)
        assert not np.array_equal(sample_image[10:40, 40:60], result[10:40, 40:60])
        assert not np.array_equal(sample_image[40:60, 10:40], result[40:60, 10:40])

    def test_empty_box_skipped(self, sample_image: np.ndarray):
        """Boxes with zero width or height should be skipped."""
        boxes = [(10, 10, 0, 30), (10, 10, 30, 0), (10, 10, 0, 0)]
        result = blur_boxes(sample_image, boxes)
        np.testing.assert_array_equal(result, sample_image)

    def test_negative_dimension_box_skipped(self, sample_image: np.ndarray):
        """Boxes with negative dimensions should be skipped."""
        boxes = [(10, 10, -10, 30), (10, 10, 30, -10)]
        result = blur_boxes(sample_image, boxes)
        np.testing.assert_array_equal(result, sample_image)

    def test_box_clamped_to_image_bounds(self, sample_image: np.ndarray):
        """Boxes extending beyond image should be clamped."""
        # Box extends beyond right and bottom edges, but starts at boundary
        boxes = [(40, 40, 80, 80)]  # Would extend to (120, 120) but clamped to (100, 100)
        result = blur_boxes(sample_image, boxes, ksize=5, sigma=5)

        # Should have blurred the valid region which crosses all 4 quadrants
        assert not np.array_equal(sample_image[40:100, 40:100], result[40:100, 40:100])

    def test_box_completely_outside_image(self, sample_image: np.ndarray):
        """Box completely outside image bounds should be skipped."""
        boxes = [(200, 200, 50, 50)]  # Completely outside 100x100 image
        result = blur_boxes(sample_image, boxes)
        np.testing.assert_array_equal(result, sample_image)

    def test_negative_coordinates_clamped(self, sample_image: np.ndarray):
        """Negative starting coordinates should be clamped to 0."""
        # Box with negative start but extends across color boundary
        boxes = [(-10, -10, 65, 65)]  # Starts at -10, -10 but clamped to 0,0 and extends past 50,50
        result = blur_boxes(sample_image, boxes, ksize=5, sigma=5)

        # Should have blurred from (0, 0) to (55, 55), crossing quadrant boundaries
        assert not np.array_equal(sample_image[0:55, 0:55], result[0:55, 0:55])

    def test_float_coordinates_converted(self, sample_image: np.ndarray):
        """Float coordinates should be converted to integers."""
        boxes = [(10.5, 10.7, 20.3, 20.9)]
        # Should not raise an error
        result = blur_boxes(sample_image, boxes, ksize=5, sigma=5)
        assert result.shape == sample_image.shape

    def test_even_kernel_size_corrected(self, sample_image: np.ndarray):
        """Even kernel size should be auto-corrected to odd."""
        boxes = [(10, 10, 30, 30)]
        # Should not raise an error even with even kernel size
        result = blur_boxes(sample_image, boxes, ksize=24, sigma=10)
        assert result.shape == sample_image.shape

    def test_default_parameters(self, sample_image: np.ndarray):
        """Test with default kernel size and sigma."""
        boxes = [(10, 10, 30, 30)]
        result = blur_boxes(sample_image, boxes)
        assert result.shape == sample_image.shape

    def test_grayscale_image(self, grayscale_image: np.ndarray):
        """Should work with grayscale (2D) images."""
        # Use a box that crosses the checkerboard boundaries
        boxes = [(40, 40, 20, 20)]  # Crosses all 4 quadrants
        result = blur_boxes(grayscale_image, boxes, ksize=5, sigma=5)
        assert result.shape == grayscale_image.shape
        assert not np.array_equal(grayscale_image[40:60, 40:60], result[40:60, 40:60])

    def test_full_image_blur(self, sample_image: np.ndarray):
        """Box covering entire image should blur everything."""
        h, w = sample_image.shape[:2]
        boxes = [(0, 0, w, h)]
        result = blur_boxes(sample_image, boxes, ksize=15, sigma=20)

        # Some difference should exist (blur smooths the quadrant boundaries)
        assert not np.array_equal(sample_image, result)


# =============================================================================
# Tests: validate_output_path
# =============================================================================


class TestValidateOutputPath:
    """Tests for the validate_output_path function."""

    def test_valid_path_returns_path_object(self, tmp_path: Path):
        """Valid path should return a Path object."""
        output = tmp_path / "output.png"
        result = validate_output_path(str(output))
        assert isinstance(result, Path)
        assert result == output

    def test_supported_extensions(self, tmp_path: Path):
        """All supported extensions should be accepted."""
        for ext in SUPPORTED_EXTENSIONS:
            output = tmp_path / f"output{ext}"
            result = validate_output_path(str(output))
            assert result.suffix.lower() == ext.lower()

    def test_unsupported_extension_raises(self, tmp_path: Path):
        """Unsupported extensions should raise SystemExit."""
        output = tmp_path / "output.gif"
        with pytest.raises(SystemExit) as exc_info:
            validate_output_path(str(output))
        assert "Unsupported output format" in str(exc_info.value)

    def test_nonexistent_directory_raises(self):
        """Non-existent parent directory should raise SystemExit."""
        output = "/nonexistent/directory/output.png"
        with pytest.raises(SystemExit) as exc_info:
            validate_output_path(output)
        assert "does not exist" in str(exc_info.value)

    def test_case_insensitive_extension(self, tmp_path: Path):
        """Extension check should be case-insensitive."""
        output = tmp_path / "output.PNG"
        result = validate_output_path(str(output))
        assert result.suffix == ".PNG"

        output = tmp_path / "output.JpEg"
        result = validate_output_path(str(output))
        assert result.suffix == ".JpEg"


# =============================================================================
# Tests: interactive_roi_selection (mocked)
# =============================================================================


class TestInteractiveRoiSelection:
    """Tests for interactive_roi_selection (with mocked OpenCV UI)."""

    def test_single_selection_then_quit(self, sample_image: np.ndarray):
        """Test selecting one ROI then quitting."""
        with patch.object(cv2, "imshow"), patch.object(cv2, "destroyAllWindows"), patch.object(
            cv2, "rectangle"
        ), patch.object(cv2, "waitKey") as mock_waitkey, patch.object(
            cv2, "selectROI"
        ) as mock_select:

            # First selectROI returns a valid box, then waitKey returns 'q'
            mock_select.return_value = (10, 20, 30, 40)
            # Pattern: waitKey(1) at start of loop, then waitKey(0) after selection
            mock_waitkey.side_effect = [1, ord("q")]

            result = roi_blur.interactive_roi_selection(sample_image)

            assert len(result) == 1
            assert result[0] == (10, 20, 30, 40)

    def test_cancelled_selection_ignored(self, sample_image: np.ndarray):
        """Test that cancelled selections (0,0,0,0) are ignored."""
        with patch.object(cv2, "imshow"), patch.object(cv2, "destroyAllWindows"), patch.object(
            cv2, "rectangle"
        ), patch.object(cv2, "waitKey") as mock_waitkey, patch.object(
            cv2, "selectROI"
        ) as mock_select:

            # Return empty selection (loop again), then valid, then quit
            mock_select.side_effect = [(0, 0, 0, 0), (10, 20, 30, 40)]
            # Pattern: waitKey(1), waitKey(0) for first iter (cancelled, continue)
            #          waitKey(1), waitKey(0) for second iter (valid, 'q' to quit)
            mock_waitkey.side_effect = [1, ord(" "), 1, ord("q")]  # space to continue, q to quit

            result = roi_blur.interactive_roi_selection(sample_image)

            assert len(result) == 1
            assert result[0] == (10, 20, 30, 40)

    def test_undo_removes_last_roi(self, sample_image: np.ndarray):
        """Test that 'u' key undoes the last ROI."""
        with patch.object(cv2, "imshow"), patch.object(cv2, "destroyAllWindows"), patch.object(
            cv2, "rectangle"
        ), patch.object(cv2, "waitKey") as mock_waitkey, patch.object(
            cv2, "selectROI"
        ) as mock_select:

            # Select first box, continue, select second box, undo second, then quit
            # After undo, the loop continues and calls selectROI again
            mock_select.side_effect = [(10, 10, 20, 20), (30, 30, 20, 20), (0, 0, 0, 0)]
            # Pattern:
            #   iter1: waitKey(1), select first box, waitKey(0) -> continue
            #   iter2: waitKey(1), select second box, waitKey(0) -> undo
            #   iter3: waitKey(1), cancelled selection, waitKey(0) -> quit
            mock_waitkey.side_effect = [1, ord(" "), 1, ord("u"), 1, ord("q")]

            result = roi_blur.interactive_roi_selection(sample_image)

            # Should only have the first ROI (second was undone)
            assert len(result) == 1
            assert result[0] == (10, 10, 20, 20)

    def test_undo_on_empty_list(self, sample_image: np.ndarray):
        """Test that undo on empty list doesn't crash."""
        with patch.object(cv2, "imshow"), patch.object(cv2, "destroyAllWindows"), patch.object(
            cv2, "rectangle"
        ), patch.object(cv2, "waitKey") as mock_waitkey, patch.object(
            cv2, "selectROI"
        ) as mock_select:

            # Cancelled selection, undo on empty, then another cancelled, then quit
            mock_select.side_effect = [(0, 0, 0, 0), (0, 0, 0, 0)]
            # Pattern: waitKey(1), cancelled, waitKey(0) -> undo
            #          waitKey(1), cancelled again, waitKey(0) -> quit
            mock_waitkey.side_effect = [1, ord("u"), 1, ord("q")]

            result = roi_blur.interactive_roi_selection(sample_image)

            assert len(result) == 0

    def test_multiple_selections_then_quit(self, sample_image: np.ndarray):
        """Test selecting multiple ROIs then quitting."""
        with patch.object(cv2, "imshow"), patch.object(cv2, "destroyAllWindows"), patch.object(
            cv2, "rectangle"
        ), patch.object(cv2, "waitKey") as mock_waitkey, patch.object(
            cv2, "selectROI"
        ) as mock_select:

            mock_select.side_effect = [(10, 20, 30, 40), (50, 60, 20, 20)]
            mock_waitkey.side_effect = [
                1,
                ord(" "),
                1,
                ord("q"),
            ]  # continue after first, quit after second

            result = roi_blur.interactive_roi_selection(sample_image)

            assert len(result) == 2
            assert result[0] == (10, 20, 30, 40)
            assert result[1] == (50, 60, 20, 20)

            assert result[1] == (50, 60, 20, 20)


# =============================================================================
# Tests: load_image
# =============================================================================


class TestLoadImage:
    """Tests for the load_image function."""

    def test_load_valid_image(self, temp_image_file: Path):
        """Should successfully load a valid image file."""
        result = roi_blur.load_image(str(temp_image_file))
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_load_nonexistent_file_raises(self):
        """Should raise SystemExit for non-existent file."""
        with pytest.raises(SystemExit) as exc_info:
            roi_blur.load_image("/nonexistent/path/image.png")
        assert (
            "not found" in str(exc_info.value).lower()
            or "does not exist" in str(exc_info.value).lower()
        )

    def test_load_invalid_image_raises(self, tmp_path: Path):
        """Should raise SystemExit for invalid image data."""
        # Create a text file with image extension
        invalid_file = tmp_path / "not_an_image.png"
        invalid_file.write_text("This is not an image")

        with pytest.raises(SystemExit) as exc_info:
            roi_blur.load_image(str(invalid_file))
        assert "failed" in str(exc_info.value).lower() or "could not" in str(exc_info.value).lower()

    def test_load_different_formats(self, tmp_path: Path, sample_image: np.ndarray):
        """Should load various image formats."""
        for ext in [".png", ".jpg", ".bmp"]:
            filepath = tmp_path / f"test{ext}"
            cv2.imwrite(str(filepath), sample_image)

            result = roi_blur.load_image(str(filepath))
            assert result is not None
            assert result.shape[:2] == sample_image.shape[:2]


# =============================================================================
# Tests: save_image
# =============================================================================


class TestSaveImage:
    """Tests for the save_image function."""

    def test_save_valid_image(self, sample_image: np.ndarray, temp_output_path: Path):
        """Should successfully save an image."""
        roi_blur.save_image(sample_image, temp_output_path)
        assert temp_output_path.exists()

        # Verify saved image can be loaded
        loaded = cv2.imread(str(temp_output_path))
        assert loaded is not None
        np.testing.assert_array_equal(loaded, sample_image)

    def test_save_creates_file(self, sample_image: np.ndarray, tmp_path: Path):
        """Should create the output file."""
        output_path = tmp_path / "new_output.png"
        assert not output_path.exists()

        roi_blur.save_image(sample_image, output_path)
        assert output_path.exists()

    def test_save_different_formats(self, sample_image: np.ndarray, tmp_path: Path):
        """Should save to different formats."""
        for ext in [".png", ".jpg", ".bmp"]:
            output_path = tmp_path / f"output{ext}"
            roi_blur.save_image(sample_image, output_path)
            assert output_path.exists()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="chmod doesn't enforce directory permissions on Windows"
    )
    def test_save_to_readonly_directory_raises(self, sample_image: np.ndarray, tmp_path: Path):
        """Should raise SystemExit when saving to read-only location fails."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        try:
            readonly_dir.chmod(0o444)
            output_path = readonly_dir / "output.png"

            with pytest.raises(SystemExit):
                roi_blur.save_image(sample_image, output_path)
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)


# =============================================================================
# Tests: parse_args
# =============================================================================


class TestParseArgs:
    """Tests for argument parsing."""

    def test_minimal_args(self, temp_image_file: Path, tmp_path: Path):
        """Should parse minimal required arguments."""
        output = tmp_path / "output.png"
        args = roi_blur.parse_args([str(temp_image_file), str(output)])

        assert args.input == str(temp_image_file)
        assert args.output == str(output)
        assert args.ksize == DEFAULT_KERNEL_SIZE
        assert args.sigma == DEFAULT_SIGMA

    def test_custom_kernel_size(self, temp_image_file: Path, tmp_path: Path):
        """Should accept custom kernel size."""
        output = tmp_path / "output.png"
        args = roi_blur.parse_args([str(temp_image_file), str(output), "--ksize", "51"])

        assert args.ksize == 51

    def test_custom_sigma(self, temp_image_file: Path, tmp_path: Path):
        """Should accept custom sigma."""
        output = tmp_path / "output.png"
        args = roi_blur.parse_args([str(temp_image_file), str(output), "--sigma", "15.5"])

        assert args.sigma == 15.5

    def test_short_options(self, temp_image_file: Path, tmp_path: Path):
        """Should accept short option forms."""
        output = tmp_path / "output.png"
        args = roi_blur.parse_args([str(temp_image_file), str(output), "-k", "31", "-s", "10"])

        assert args.ksize == 31
        assert args.sigma == 10.0

    def test_missing_input_raises(self, tmp_path: Path):
        """Should raise error when input is missing."""
        output = tmp_path / "output.png"
        with pytest.raises(SystemExit):
            roi_blur.parse_args([str(output)])

    def test_missing_output_raises(self, temp_image_file: Path):
        """Should raise error when output is missing."""
        with pytest.raises(SystemExit):
            roi_blur.parse_args([str(temp_image_file)])

    def test_negative_ksize_rejected(self, temp_image_file: Path, tmp_path: Path):
        """Should reject negative kernel size."""
        output = tmp_path / "output.png"
        with pytest.raises(SystemExit):
            roi_blur.parse_args([str(temp_image_file), str(output), "--ksize", "-5"])

    def test_zero_ksize_rejected(self, temp_image_file: Path, tmp_path: Path):
        """Should reject zero kernel size."""
        output = tmp_path / "output.png"
        with pytest.raises(SystemExit):
            roi_blur.parse_args([str(temp_image_file), str(output), "--ksize", "0"])

    def test_negative_sigma_rejected(self, temp_image_file: Path, tmp_path: Path):
        """Should reject negative sigma."""
        output = tmp_path / "output.png"
        with pytest.raises(SystemExit):
            roi_blur.parse_args([str(temp_image_file), str(output), "--sigma", "-1.0"])


# =============================================================================
# Tests: Edge cases and integration
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_very_small_image(self):
        """Should handle very small images (1x1)."""
        tiny_image = np.array([[[255, 255, 255]]], dtype=np.uint8)
        boxes = [(0, 0, 1, 1)]
        result = blur_boxes(tiny_image, boxes, ksize=3, sigma=1)
        assert result.shape == (1, 1, 3)

    def test_very_large_kernel(self, sample_image: np.ndarray):
        """Should handle kernel larger than ROI."""
        boxes = [(40, 40, 10, 10)]  # 10x10 ROI
        result = blur_boxes(sample_image, boxes, ksize=51, sigma=20)  # 51x51 kernel
        assert result.shape == sample_image.shape

    def test_zero_sigma(self, sample_image: np.ndarray):
        """Zero sigma should result in minimal/no blur."""
        boxes = [(10, 10, 30, 30)]
        result = blur_boxes(sample_image, boxes, ksize=5, sigma=0)
        # With sigma=0, OpenCV calculates sigma from kernel size
        assert result.shape == sample_image.shape

    def test_very_high_sigma(self, sample_image: np.ndarray):
        """Very high sigma should produce heavy blur."""
        boxes = [(10, 10, 30, 30)]
        result = blur_boxes(sample_image, boxes, ksize=31, sigma=100)
        assert result.shape == sample_image.shape

    def test_overlapping_boxes(self, sample_image: np.ndarray):
        """Overlapping boxes should both be processed."""
        boxes = [(10, 10, 40, 40), (30, 30, 40, 40)]
        result = blur_boxes(sample_image, boxes, ksize=11, sigma=10)

        # Overlapping region should be doubly blurred
        assert not np.array_equal(sample_image, result)

    def test_rgba_image(self):
        """Should handle RGBA (4-channel) images."""
        rgba_image = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba_image[:, :, 3] = 255  # Full opacity
        rgba_image[25:75, 25:75, 0] = 255  # Red square

        boxes = [(25, 25, 50, 50)]
        result = blur_boxes(rgba_image, boxes, ksize=5, sigma=5)

        assert result.shape == rgba_image.shape
        assert result.shape[2] == 4  # Still 4 channels

    def test_16bit_image(self):
        """Should handle 16-bit images."""
        img_16bit = np.zeros((100, 100, 3), dtype=np.uint16)
        img_16bit[25:75, 25:75] = 65535  # Max 16-bit value

        boxes = [(25, 25, 50, 50)]
        result = blur_boxes(img_16bit, boxes, ksize=5, sigma=5)

        assert result.dtype == np.uint16

    def test_float_image(self):
        """Should handle float images."""
        img_float = np.zeros((100, 100, 3), dtype=np.float32)
        img_float[25:75, 25:75] = 1.0

        boxes = [(25, 25, 50, 50)]
        result = blur_boxes(img_float, boxes, ksize=5, sigma=5)

        assert result.dtype == np.float32


# =============================================================================
# Tests: Command-line integration
# =============================================================================


class TestCLIIntegration:
    """Integration tests for command-line interface."""

    def test_cli_basic_execution(self, temp_image_file: Path, tmp_path: Path):
        """Test basic CLI execution with mocked interactive selection."""
        output_path = tmp_path / "cli_output.png"

        with patch.object(roi_blur, "interactive_roi_selection") as mock_select, patch.object(
            cv2, "imshow"
        ), patch.object(cv2, "waitKey", return_value=0), patch.object(cv2, "destroyAllWindows"):
            mock_select.return_value = [(10, 10, 30, 30)]

            # Run main with arguments
            try:
                roi_blur.main([str(temp_image_file), str(output_path)])
            except SystemExit as e:
                # main() might call sys.exit(0) on success
                if e.code != 0 and e.code is not None:
                    raise

        assert output_path.exists()

    def test_cli_no_rois_selected(self, temp_image_file: Path, tmp_path: Path):
        """Test CLI when no ROIs are selected."""
        output_path = tmp_path / "cli_output.png"

        with patch.object(roi_blur, "interactive_roi_selection") as mock_select, patch.object(
            cv2, "imshow"
        ), patch.object(cv2, "waitKey", return_value=0), patch.object(cv2, "destroyAllWindows"):
            mock_select.return_value = []  # No ROIs selected

            # Should complete and save unmodified image
            with contextlib.suppress(SystemExit):
                roi_blur.main([str(temp_image_file), str(output_path)])

        # Output should still be created (with unmodified image)
        assert output_path.exists()

    def test_subprocess_execution(self):
        """Test running the script as a subprocess."""
        # Skip if roi_blur.py doesn't exist as a standalone file
        script_path = Path(roi_blur.__file__)
        if not script_path.exists():
            pytest.skip("Script file not found for subprocess test")

        # This test would require a way to mock the interactive selection
        # in a subprocess, which is complex. Mark as expected to need user input.
        pytest.skip("Subprocess test requires interactive input")


# =============================================================================
# Tests: Performance (optional, marked slow)
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked slow, can be skipped with -m 'not slow')."""

    def test_large_image_performance(self):
        """Test performance with a large image."""
        # 4K image
        large_image = np.random.randint(0, 256, (2160, 3840, 3), dtype=np.uint8)
        boxes = [(100, 100, 500, 500), (1000, 1000, 800, 800)]

        import time

        start = time.time()
        result = blur_boxes(large_image, boxes, ksize=31, sigma=15)
        elapsed = time.time() - start

        assert result.shape == large_image.shape
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_many_boxes_performance(self, sample_image: np.ndarray):
        """Test performance with many small boxes."""
        # Create a grid of small boxes
        boxes = [(x, y, 8, 8) for x in range(0, 100, 10) for y in range(0, 100, 10)]  # 100 boxes

        import time

        start = time.time()
        result = blur_boxes(sample_image, boxes, ksize=5, sigma=3)
        elapsed = time.time() - start

        assert result.shape == sample_image.shape
        assert elapsed < 2.0  # Should complete quickly


# =============================================================================
# Tests: Blur quality verification
# =============================================================================


class TestBlurQuality:
    """Tests for blur quality verification."""

    def test_blur_smooths_edges(self, sample_image: np.ndarray):
        """Blur should smooth sharp edges in the selected region."""
        boxes = [(10, 10, 80, 80)]
        result = blur_boxes(sample_image, boxes, ksize=21, sigma=15)

        # The blurred region should have smoothed transitions
        assert not np.array_equal(sample_image, result)

    def test_blur_preserves_image_dtype(self, sample_image: np.ndarray):
        """Blur should preserve the image data type."""
        boxes = [(10, 10, 30, 30)]
        result = blur_boxes(sample_image, boxes)

        assert result.dtype == sample_image.dtype
