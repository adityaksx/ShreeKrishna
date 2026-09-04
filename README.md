# Animated Image Sketch & Color Fill

This Python project converts an input image into an animated drawing:

1. Loads the source image.
2. Separates the foreground from the background.
3. Smooths and quantizes the image colors.
4. Detects dark ink/outline regions.
5. Animates the outline drawing.
6. Fills the artwork with posterized colors.

The project uses **OpenCV**, **NumPy**, and **Pygame**.

## Requirements

- Python 3.10+ recommended
- OpenCV
- NumPy
- Pygame

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

Keep the files in the same folder:

```text
project/
├── main.py
├── radha.jpg
├── requirements.txt
└── README.md
```

The program currently expects the input image to be named:

```text
radha.jpg
```

If you want to use another image, either rename it to `radha.jpg` or change this line in `main.py`:

```python
IMAGE_PATH = "radha.jpg"
```

## Run

Open a terminal inside the project directory and run:

```bash
python main.py
```

On some Linux systems you may need:

```bash
python3 main.py
```

## Controls

While the animation window is open:

- `SPACE` — Pause / resume
- `R` — Restart the animation
- `ESC` — Exit
- Closing the Pygame window also exits the program

## How It Works

The program first performs all image processing before opening the animation window.

### 1. Foreground Detection

The colors in the four corners of the image are sampled to estimate the background color.

Pixels sufficiently different from that background are treated as foreground.

The main parameter controlling this is:

```python
BACKGROUND_DISTANCE = 32
```

Lower values include more pixels in the foreground.

Higher values remove more pixels that resemble the background.

### 2. Edge-Preserving Smoothing

The image is processed with OpenCV mean-shift filtering:

```python
cv2.pyrMeanShiftFiltering(...)
```

This reduces small color variations while preserving important boundaries and artwork lines.

Relevant parameters:

```python
MEANSHIFT_SPATIAL_RADIUS = 10
MEANSHIFT_COLOR_RADIUS = 24
```

### 3. Color Quantization

K-means clustering reduces the image to a smaller color palette:

```python
COLOR_CLUSTERS = 16
```

Increasing this value preserves more colors but can create more fragmented regions.

Decreasing it creates a simpler, flatter cartoon-style result.

### 4. Ink / Outline Detection

The program assumes that the darkest color clusters represent the original artwork linework.

The main setting is:

```python
INK_VALUE_THRESHOLD = 55
```

If very few or no outlines appear, try increasing this value.

If too many dark regions are treated as outlines, reduce it.

### 5. Outline Animation

Contours from the detected ink mask are converted into paths and drawn gradually.

Animation speed is controlled by:

```python
OUTLINE_POINTS_PER_FRAME = 6
```

Increase it for faster outline drawing.

Decrease it for slower outline drawing.

### 6. Color Animation

The detected color regions are converted into horizontal fill strokes.

Color fill speed is controlled by:

```python
COLOR_STROKES_PER_FRAME = 10
```

Increase it for faster coloring.

Decrease it for slower coloring.

## Important Configuration

The main settings are near the top of `main.py`:

```python
IMAGE_PATH = "radha.jpg"

WIDTH = 1000
HEIGHT = 1000
FPS = 60

MAX_IMAGE_SIZE = 900

BACKGROUND_DISTANCE = 32

MEANSHIFT_SPATIAL_RADIUS = 10
MEANSHIFT_COLOR_RADIUS = 24
COLOR_CLUSTERS = 16

INK_VALUE_THRESHOLD = 55

MIN_COLOR_REGION_AREA = 40
MAX_COLOR_REGIONS = 400

OUTLINE_POINTS_PER_FRAME = 6
COLOR_STROKES_PER_FRAME = 10
```

## If No Outline Is Detected

If the terminal prints:

```text
ERROR: Zero outline paths were produced.
```

try adjusting:

```python
INK_VALUE_THRESHOLD
```

or:

```python
BACKGROUND_DISTANCE
```

For example:

```python
INK_VALUE_THRESHOLD = 70
```

can help when the original image contains dark gray outlines instead of nearly black outlines.

## Recommended Images

The algorithm works best with:

- cartoon or vector-style artwork
- clearly visible dark outlines
- relatively simple or flat backgrounds
- strong separation between the subject and background
- reasonably large source images
- limited motion blur or compression artifacts

Highly realistic photographs, noisy backgrounds, very soft shading, or images without distinct dark outlines may produce weaker results.

## Linux / Fedora

Install Python and pip if required:

```bash
sudo dnf install python3 python3-pip
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

If Pygame prints an AVX2 warning, it is normally a performance warning rather than a program-breaking error.

## License

You can add your preferred license before publishing the project publicly.

For example, if you want an open-source project, consider adding an MIT License file.
