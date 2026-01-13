# ROI Blur Tool

A simple, interactive OpenCV-based utility for selecting regions of interest (ROIs) in an image and applying Gaussian blur to those regions. Perfect for privacy redaction, hiding sensitive information, or creative effects.

## Features

- **Interactive Selection**: Click and drag to draw ROI rectangles
- **Undo Support**: Remove accidentally drawn ROIs with 'u' key
- **Adjustable Blur**: Control kernel size and sigma via CLI
- **Multiple Formats**: Supports JPG, PNG, BMP, TIFF, WebP
- **Color Preservation**: Maintains ICC color profiles and metadata
- **Robust**: Handles edge cases, validates inputs, clamps to bounds

## Installation

### Prerequisites

- Python 3.8+
- OpenCV 4.x

### Install via pip

```shell
# Clone the repository
git clone https://github.com/techquestsdev/roi-blur.git
cd roi-blur

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install via Poetry

```shell
git clone https://github.com/techquestsdev/roi-blur.git
cd roi-blur
poetry install
```

## Usage

### Basic Usage

```shell
python roi_blur.py input.jpg output.jpg
```

### With Custom Blur Settings

```shell
python roi_blur.py photo.png blurred.png --ksize 51 --sigma 50
```

### Interactive Controls

| Key | Action |
|-----|--------|
|Click + Drag | Draw ROI rectangle |
|ENTER / SPACE | Confirm selection |
|ESC | Cancel current selection |
|u | Undo last ROI |
|q | Finish and apply blur |

```shell
usage: roi_blur [-h] [-k N] [-s N] [-v] INPUT OUTPUT

Interactively select regions in an image and apply Gaussian blur.

positional arguments:
  INPUT                 Path to the input image file
  OUTPUT                Path for the output image file

options:
  -h, --help            show this help message and exit
  -k N, --ksize N       Blur kernel size (positive odd integer, default: 23)
  -s N, --sigma N       Blur sigma/strength (positive float, default: 30.0)
  -v, --version         show program's version number and exit
```

## Examples

### Blur Faces for Privacy

```shell
python roi_blur.py family_photo.jpg privacy_safe.jpg --ksize 45 --sigma 60
```

### Redact Sensitive Text

```shell
python roi_blur.py document.png redacted.png --ksize 31 --sigma 40
```

### Artistic Background Blur

```shell
python roi_blur.py portrait.jpg artistic.jpg --ksize 15 --sigma 20
```

## Programmatic Usage

You can also use the blur function programmatically:

```python
import cv2
from roi_blur import blur_boxes

# Load image
image = cv2.imread("photo.jpg")

# Define ROIs: list of (x, y, width, height) tuples
boxes = [
    (100, 100, 200, 150),  # First region
    (400, 300, 100, 100),  # Second region
]

# Apply blur
result = blur_boxes(image, boxes, ksize=31, sigma=40)

# Save result
cv2.imwrite("blurred.jpg", result)
```

## How It Works

1. **Load Image**: OpenCV reads the input image into a NumPy array
2. **ROI Selection**: User draws rectangles using OpenCV's selectROI
3. **Gaussian Blur**: Each selected region is extracted, blurred, and replaced
4. **Output**: Result is displayed and optionally saved to disk

### Technical Details

- Uses Pillow for loading/saving to preserve ICC color profiles
- Uses `cv2.GaussianBlur` with `BORDER_REPLICATE` to avoid edge artifacts
- Kernel size is automatically adjusted to be odd (OpenCV requirement)
- ROI coordinates are clamped to image bounds for safety
- Original image is never modified (copy-on-write pattern)

## Development

### Running Tests

```shell
pytest tests/ -v
```

### Code Quality

```shell
# Linting
ruff check roi_blur.py

# Type checking
mypy roi_blur.py

# Formatting
black roi_blur.py
```

### Project Structure

```txt
roi-blur/
├── roi_blur.py          # Main application
├── requirements.txt     # Dependencies
├── pyproject.toml       # Project configuration
├── tests/
│   └── test_roi_blur.py # Unit tests
├── LICENSE              # GPL-3.0
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for the computer vision library
- [NumPy](https://numpy.org/) for array operations
- [Pillow](https://python-pillow.org/) for image I/O with color profile support

---

Made with ❤️ and Python
