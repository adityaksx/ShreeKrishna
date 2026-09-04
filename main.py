
import cv2
import numpy as np
import pygame


# ============================================================
# IMAGE -> PRECOMPUTED OUTLINE -> ANIMATED OUTLINE -> COLOR
#
# v3 CHANGES (fixes "too much outline noise" / "outline doesn't
# blend with color" from v2):
#
#   v2 traced a boundary line between EVERY pair of adjacent
#   quantized clusters. Flat art has soft shading gradients, and
#   even after smoothing, kmeans still splits a gradient into
#   several similar clusters -- so v2 drew hundreds of tiny
#   boundary loops inside what should be one smooth shape. That's
#   the speckle/noise you saw, and since those lines were drawn
#   in flat WHITE on top of the fill, they never blended in.
#
#   v3 doesn't trace edges at all. Cartoon/vector art like this
#   already has its linework baked in as literal dark pixels. So:
#     1. cv2.pyrMeanShiftFiltering first -- an edge-preserving
#        smoothing that flattens soft shading gradients into flat
#        blocks WITHOUT blurring away the real ink lines. This is
#        what actually kills the speckle (kmeans on a flattened
#        image no longer flip-flops between near-identical
#        clusters pixel to pixel).
#     2. kmeans on top of that to get a small clean palette.
#     3. The darkest cluster(s) in that palette ARE the ink lines
#        -- no detection needed, just a brightness threshold.
#   The white "sketch" animation phase traces that real ink mask
#   (clean, since it's an actual shape, not a noise pattern).
#   The color phase then fills every cluster -- INCLUDING the ink
#   cluster, in its real near-black color -- so the line work
#   ends up rendered in its own true color as part of the same
#   fill pass, instead of a mismatched white overlay on top.
# ============================================================

IMAGE_PATH = "radha.jpg"

WIDTH = 1000
HEIGHT = 1000
FPS = 60

MAX_IMAGE_SIZE = 900

# --- Foreground / background separation ---
BACKGROUND_DISTANCE = 32

# --- Color palette / posterization ---
MEANSHIFT_SPATIAL_RADIUS = 10
MEANSHIFT_COLOR_RADIUS = 24
COLOR_CLUSTERS = 16

# A cluster whose color's brightest channel is below this value
# is treated as "ink" (line work) rather than a fill color.
INK_VALUE_THRESHOLD = 55

MIN_COLOR_REGION_AREA = 40
MAX_COLOR_REGIONS = 400
FILL_ROW_STEP = 3   # thickness (px) of each gapless fill band

# --- Outline path extraction (applied to the ink mask only) ---
MIN_CONTOUR_LENGTH = 8
MAX_OUTLINE_PATHS = 2500
LINE_APPROX = 0.0015

# Slow drawing animation
OUTLINE_POINTS_PER_FRAME = 6
COLOR_STROKES_PER_FRAME = 10

BLACK = (0, 0, 0)
WHITE = (250, 250, 247)
SOFT_WHITE = (215, 215, 210)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):
    bgr = cv2.imread(path)

    if bgr is None:
        raise FileNotFoundError(
            f"Could not find {path!r}. "
            "Put radha.jpg in the same folder as main.py."
        )

    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIZE / max(h, w))

    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# FIT IMAGE
# ============================================================

def fit_image(rgb):
    h, w = rgb.shape[:2]

    scale = min((WIDTH - 100) / w, (HEIGHT - 100) / h)

    nw = int(w * scale)
    nh = int(h * scale)

    fitted = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    ox = (WIDTH - nw) // 2
    oy = (HEIGHT - nh) // 2

    return fitted, ox, oy


# ============================================================
# FOREGROUND MASK
#
# Distance from the sampled background color. Robust to any flat
# background, and correctly excludes faint drop-shadows baked
# into the art instead of treating them as an extra solid shape.
# ============================================================

def make_foreground_mask(rgb):
    h, w = rgb.shape[:2]
    corner_size = max(4, min(h, w) // 20)

    samples = np.concatenate([
        rgb[:corner_size, :corner_size].reshape(-1, 3),
        rgb[:corner_size, -corner_size:].reshape(-1, 3),
        rgb[-corner_size:, :corner_size].reshape(-1, 3),
        rgb[-corner_size:, -corner_size:].reshape(-1, 3),
    ]).astype(np.float32)

    bg_color = samples.mean(axis=0)

    dist = np.linalg.norm(rgb.astype(np.float32) - bg_color, axis=2)

    mask = (dist > BACKGROUND_DISTANCE).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > 200:
            cv2.drawContours(filled, [c], -1, 255, -1)

    return filled


# ============================================================
# COLOR QUANTIZATION (posterize, then palette-reduce)
# ============================================================

def quantize_colors(rgb, foreground):
    # Edge-preserving smoothing: flattens soft shading gradients
    # into solid blocks but keeps real ink lines sharp. This is
    # what prevents kmeans from fragmenting a gradient into many
    # tiny alternating-cluster speckles.
    flattened = cv2.pyrMeanShiftFiltering(
        rgb,
        sp=MEANSHIFT_SPATIAL_RADIUS,
        sr=MEANSHIFT_COLOR_RADIUS
    )

    usable = foreground > 0
    pixels = flattened[usable]

    if len(pixels) < COLOR_CLUSTERS:
        return rgb.copy(), np.array([[128, 128, 128]], dtype=np.uint8)

    samples = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        samples, COLOR_CLUSTERS, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(np.clip(centers, 0, 255))
    labels = labels.flatten()

    quantized = np.zeros_like(rgb)
    ys, xs = np.where(usable)
    quantized[ys, xs] = centers[labels]

    return quantized, centers


def find_ink_mask(quantized, centers, foreground):
    """The darkest clusters ARE the artwork's linework -- no edge
    detection needed, just pick them out by brightness."""

    ink_mask = np.zeros(quantized.shape[:2], np.uint8)

    for center in centers:
        if int(center.max()) < INK_VALUE_THRESHOLD:
            m = np.all(quantized == center, axis=2)
            ink_mask[m] = 255

    return cv2.bitwise_and(ink_mask, foreground)


# ============================================================
# PRECOMPUTE ANIMATED OUTLINE PATHS (from the real ink mask)
# ============================================================

def precompute_outline(ink_mask):
    contours, _ = cv2.findContours(
        ink_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    candidates = []

    for contour in contours:
        length = cv2.arcLength(contour, False)

        if length < MIN_CONTOUR_LENGTH:
            continue

        epsilon = max(0.45, length * LINE_APPROX)
        approx = cv2.approxPolyDP(contour, epsilon, False).reshape(-1, 2)

        if len(approx) < 2:
            continue

        candidates.append({
            "points": approx,
            "length": float(length),
            "bottom": float(np.max(approx[:, 1])),
            "mean_y": float(np.mean(approx[:, 1]))
        })

    candidates.sort(key=lambda x: x["length"], reverse=True)
    candidates = candidates[:MAX_OUTLINE_PATHS]

    if not candidates:
        return []

    bottom = max(p["bottom"] for p in candidates)
    top = min(float(np.min(p["points"][:, 1])) for p in candidates)
    span = max(1.0, bottom - top)

    BAND_COUNT = 30

    for p in candidates:
        normalized = (bottom - p["bottom"]) / span
        p["band"] = max(0, min(BAND_COUNT - 1, int(normalized * BAND_COUNT)))

    candidates.sort(key=lambda p: (p["band"], -p["length"], p["mean_y"]))

    return [
        [(int(x), int(y)) for x, y in item["points"]]
        for item in candidates
    ]


# ============================================================
# PRECOMPUTE COLOR REGIONS (every cluster, ink included)
# ============================================================

def precompute_color_regions(quantized, foreground, centers):
    regions = []

    for center in centers:
        region_mask = np.all(quantized == center, axis=2).astype(np.uint8) * 255
        region_mask = cv2.bitwise_and(region_mask, foreground)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            region_mask, 8
        )

        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])

            if area < MIN_COLOR_REGION_AREA:
                continue

            regions.append({
                "mask": (labels == component),
                "color": tuple(int(v) for v in center),
                "area": area,
                "center_y": float(centroids[component][1])
            })

    regions.sort(key=lambda r: r["area"], reverse=True)
    regions = regions[:MAX_COLOR_REGIONS]

    regions.sort(key=lambda r: (-r["center_y"], -r["area"]))

    return regions


# ============================================================
# PRECOMPUTE COLOR FILL STROKES (gapless bands, bottom -> top)
# ============================================================

def make_paint_queue(regions, ox, oy):
    queue = []

    for region in regions:
        mask = region["mask"]
        ys = np.where(mask.any(axis=1))[0]

        if len(ys) == 0:
            continue

        y_min = int(ys.min())
        y_max = int(ys.max())

        strokes = []

        for y in range(y_max, y_min - 1, -FILL_ROW_STEP):
            y0 = max(y_min, y - FILL_ROW_STEP + 1)
            band = mask[y0:y + 1]

            if band.size == 0:
                continue

            col_any = band.any(axis=0)
            xs = np.where(col_any)[0]

            if len(xs) == 0:
                continue

            breaks = np.where(np.diff(xs) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, len(xs) - 1]

            mid_y = (y0 + y) // 2

            for s, e in zip(starts, ends):
                x1 = int(xs[s])
                x2 = int(xs[e])

                strokes.append({
                    "a": (int(ox + x1), int(oy + mid_y)),
                    "b": (int(ox + x2), int(oy + mid_y)),
                    "width": (y - y0) + FILL_ROW_STEP
                })

        if strokes:
            queue.append({"color": region["color"], "strokes": strokes})

    return queue


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # STAGE 1: FULL INTERNAL PREPARATION
    # ========================================================

    print("Loading image...")
    original = load_image(IMAGE_PATH)
    fitted, ox, oy = fit_image(original)

    print("Preparing foreground...")
    foreground = make_foreground_mask(fitted)

    print("Posterizing and quantizing colors...")
    quantized, centers = quantize_colors(fitted, foreground)

    print("Extracting real ink lines...")
    ink_mask = find_ink_mask(quantized, centers, foreground)

    print("Precomputing animated outline paths...")
    paths_local = precompute_outline(ink_mask)
    paths = [[(x + ox, y + oy) for x, y in path] for path in paths_local]

    print(f"Prepared {len(paths)} outline paths.")

    if not paths:
        print("\nERROR: Zero outline paths were produced.")
        print(
            "Try lowering INK_VALUE_THRESHOLD or BACKGROUND_DISTANCE."
        )
        return

    regions = precompute_color_regions(quantized, foreground, centers)
    print(f"Prepared {len(regions)} color regions.")

    print("Precomputing complete color fill...")
    paint_queue = make_paint_queue(regions, ox, oy)

    total_color_strokes = sum(len(r["strokes"]) for r in paint_queue)
    print(f"Prepared {total_color_strokes} color fill strokes.")

    print("\nALL PROCESSING COMPLETE.")
    print("Starting visible drawing animation...")

    # ========================================================
    # STAGE 2: VISIBLE ANIMATION
    # ========================================================

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Animated Sketch")
    clock = pygame.time.Clock()

    outline_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    color_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    phase = "OUTLINE"
    path_index = 0
    point_index = 1
    region_index = 0
    stroke_index = 0

    paused = False
    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    phase = "OUTLINE"
                    path_index = 0
                    point_index = 1
                    region_index = 0
                    stroke_index = 0
                    outline_layer.fill((0, 0, 0, 0))
                    color_layer.fill((0, 0, 0, 0))
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # ----------------------------------------------------
        # PHASE 1: SHOW PRECOMPUTED WHITE SKETCH (the real ink)
        # ----------------------------------------------------

        if not paused and phase == "OUTLINE":

            for _ in range(OUTLINE_POINTS_PER_FRAME):

                if path_index >= len(paths):
                    phase = "COLOR"
                    break

                path = paths[path_index]

                if point_index >= len(path):
                    path_index += 1
                    point_index = 1
                    continue

                a = path[point_index - 1]
                b = path[point_index]

                color = WHITE if len(path) >= 8 else SOFT_WHITE
                pygame.draw.line(outline_layer, color, a, b, 2)

                point_index += 1

        # ----------------------------------------------------
        # PHASE 2: FILL COLOR (ink cluster included, in its real
        # dark color -- this is what makes it "blend": the line
        # work ends up rendered in its own true color as part of
        # the same fill pass, not a separate overlay).
        # ----------------------------------------------------

        elif not paused and phase == "COLOR":

            for _ in range(COLOR_STROKES_PER_FRAME):

                if region_index >= len(paint_queue):
                    phase = "DONE"
                    break

                region = paint_queue[region_index]
                strokes = region["strokes"]

                if stroke_index >= len(strokes):
                    region_index += 1
                    stroke_index = 0
                    continue

                stroke = strokes[stroke_index]

                color = tuple(
                    min(255, int(v * 1.03)) for v in region["color"]
                ) + (255,)

                pygame.draw.line(
                    color_layer, color, stroke["a"], stroke["b"], stroke["width"]
                )

                stroke_index += 1

        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        screen.fill(BLACK)
        screen.blit(outline_layer, (0, 0))
        screen.blit(color_layer, (0, 0))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
