"""
Generate synthetic X-ray test images.

Creates 6 grayscale images that visually resemble chest X-rays,
each representing a different clinical scenario for testing.

Usage:
    pip install pillow numpy
    python generate_samples.py

Output: 6 PNG files in the current directory (sample_xrays/)
"""

import math
import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
except ImportError:
    print("Install dependencies first:  pip install pillow numpy")
    sys.exit(1)

SAMPLES = [
    {
        "filename": "torax_normal_pa.png",
        "label": "Tórax PA Normal",
        "desc": "Raio-X de tórax em posição PA com campos pulmonares limpos",
        "pathology": None,
    },
    {
        "filename": "torax_pneumonia_lobo_inferior.png",
        "label": "Pneumonia Lobo Inferior",
        "desc": "Consolidação em lobo inferior direito sugestiva de pneumonia",
        "pathology": "pneumonia",
    },
    {
        "filename": "torax_derrame_pleural.png",
        "label": "Derrame Pleural",
        "desc": "Opacidade homogênea em base esquerda com apagamento do seio costofrênico",
        "pathology": "pleural_effusion",
    },
    {
        "filename": "coluna_fratura_vertebral.png",
        "label": "Fratura Vertebral",
        "desc": "Coluna torácica lateral com colapso vertebral em T8",
        "pathology": "fracture",
    },
    {
        "filename": "torax_cardiomegalia.png",
        "label": "Cardiomegalia",
        "desc": "Tórax PA com índice cardiotorácico aumentado",
        "pathology": "cardiomegaly",
    },
    {
        "filename": "torax_nodulo_pulmonar.png",
        "label": "Nódulo Pulmonar",
        "desc": "Nódulo solitário em terço médio do pulmão direito",
        "pathology": "nodule",
    },
]

W, H = 512, 512
RNG  = random.Random(42)
NP_RNG = np.random.default_rng(42)


def make_base_chest(width=W, height=H) -> np.ndarray:
    """Create a grayscale array resembling a chest X-ray."""
    img = np.zeros((height, width), dtype=np.float32)
    cx, cy = width // 2, height // 2

    # Background body silhouette
    for y in range(height):
        for x in range(width):
            dx = (x - cx) / (width * 0.42)
            dy = (y - cy) / (height * 0.48)
            if dx*dx + dy*dy < 1.0:
                img[y, x] = 0.18

    # Lung fields (dark = air-filled)
    for y in range(height):
        for x in range(width):
            # Left lung
            lx = (x - cx * 0.58) / (width * 0.20)
            ly = (y - cy * 0.90) / (height * 0.34)
            if lx*lx + ly*ly < 1.0:
                img[y, x] = max(0.03, img[y, x] - 0.12)
            # Right lung
            rx = (x - cx * 1.40) / (width * 0.20)
            ry = (y - cy * 0.90) / (height * 0.34)
            if rx*rx + ry*ry < 1.0:
                img[y, x] = max(0.03, img[y, x] - 0.12)

    # Spine (bright vertical band)
    for y in range(int(height * 0.12), int(height * 0.88)):
        for x in range(int(cx * 0.90), int(cx * 1.10)):
            img[y, x] = min(1.0, img[y, x] + 0.35)

    # Ribs — horizontal arcs
    for rib in range(8):
        y_center = int(height * (0.18 + rib * 0.09))
        thickness = 5
        for y in range(max(0, y_center - thickness), min(height, y_center + thickness)):
            dy = abs(y - y_center) / thickness
            brightness = 0.50 * (1 - dy)
            for x in range(int(width * 0.12), int(width * 0.88)):
                angle = math.pi * (x - cx) / (width * 0.38)
                arc_y = int(y_center + 18 * math.sin(angle))
                if abs(y - arc_y) < thickness:
                    img[y, x] = min(1.0, img[y, x] + brightness)

    # Diaphragm domes
    for x in range(int(width * 0.1), int(width * 0.9)):
        dome_h = int(height * 0.72 - 22 * math.cos(math.pi * (x - cx) / (width * 0.38)))
        for dy in range(-4, 5):
            y = dome_h + dy
            if 0 <= y < height:
                img[y, x] = min(1.0, img[y, x] + 0.30 * (1 - abs(dy) / 5))

    # Heart shadow (left mid-lower)
    for y in range(height):
        for x in range(width):
            hx = (x - cx * 0.80) / (width * 0.14)
            hy = (y - cy * 1.05) / (height * 0.18)
            if hx*hx + hy*hy < 1.0:
                img[y, x] = min(0.65, img[y, x] + 0.20)

    # Add subtle lung vasculature texture
    vascular = NP_RNG.normal(0, 0.025, (height, width)).astype(np.float32)
    img = np.clip(img + vascular, 0, 1)

    return img


def add_consolidation(img: np.ndarray, side: str = "right") -> np.ndarray:
    """Add a patchy consolidation (pneumonia-like) to lower lobe."""
    cx, cy = W // 2, H // 2
    if side == "right":
        ox, oy = int(cx * 1.35), int(cy * 1.25)
    else:
        ox, oy = int(cx * 0.65), int(cy * 1.25)

    arr = img.copy()
    for y in range(H):
        for x in range(W):
            dx = (x - ox) / 50
            dy = (y - oy) / 40
            if dx*dx + dy*dy < 1.0:
                noise = NP_RNG.normal(0, 0.04)
                arr[y, x] = min(1.0, arr[y, x] + 0.30 + noise)
    return arr


def add_pleural_effusion(img: np.ndarray, side: str = "left") -> np.ndarray:
    """Add blunting of costophrenic angle (pleural effusion)."""
    arr = img.copy()
    fill_y = int(H * 0.76)
    if side == "left":
        x_start, x_end = int(W * 0.10), int(W * 0.46)
    else:
        x_start, x_end = int(W * 0.54), int(W * 0.90)

    for y in range(fill_y, H):
        for x in range(x_start, x_end):
            depth = (y - fill_y) / (H - fill_y)
            arr[y, x] = min(1.0, arr[y, x] + 0.45 * depth)
    return arr


def add_cardiomegaly(img: np.ndarray) -> np.ndarray:
    """Enlarge the cardiac silhouette."""
    cx, cy = W // 2, H // 2
    arr = img.copy()
    for y in range(H):
        for x in range(W):
            hx = (x - cx * 0.80) / (width_scale := W * 0.22)
            hy = (y - cy * 1.05) / (H * 0.26)
            if hx*hx + hy*hy < 1.0:
                arr[y, x] = min(0.75, arr[y, x] + 0.25)
    return arr


def add_nodule(img: np.ndarray) -> np.ndarray:
    """Add a solitary pulmonary nodule."""
    arr = img.copy()
    nx, ny = int(W * 0.68), int(H * 0.42)
    r = 14
    for y in range(max(0, ny - r), min(H, ny + r)):
        for x in range(max(0, nx - r), min(W, nx + r)):
            dx = (x - nx) / r
            dy = (y - ny) / r
            if dx*dx + dy*dy < 1.0:
                fade = 1 - math.sqrt(dx*dx + dy*dy)
                arr[y, x] = min(1.0, arr[y, x] + 0.55 * fade)
    return arr


def add_vertebral_fracture(img: np.ndarray) -> np.ndarray:
    """Add collapse of a vertebral body on lateral spine view."""
    arr = np.zeros((H, W), dtype=np.float32)
    cx = W // 2

    # Vertebral column (lateral view)
    for y in range(int(H * 0.10), int(H * 0.90)):
        for x in range(int(cx * 0.70), int(cx * 1.30)):
            arr[y, x] = 0.45

    # Vertebral bodies
    for v in range(12):
        vy = int(H * (0.12 + v * 0.065))
        height_v = 28 if v != 7 else 14  # T8 compressed
        for y in range(vy, min(H, vy + height_v)):
            for x in range(int(cx * 0.78), int(cx * 1.22)):
                arr[y, x] = 0.75

    # Anterior wedging of T8
    vy8 = int(H * (0.12 + 7 * 0.065))
    for y in range(vy8, min(H, vy8 + 14)):
        taper = (y - vy8) / 14
        x_start = int(cx * 0.78)
        x_end   = int(cx * (0.78 + 0.44 * (0.4 + 0.6 * taper)))
        for x in range(x_start, x_end):
            arr[y, x] = 0.80

    vascular = NP_RNG.normal(0, 0.02, (H, W)).astype(np.float32)
    return np.clip(arr + vascular, 0, 1)


def to_pil(arr: np.ndarray) -> Image.Image:
    """Convert float32 [0,1] array to PIL Image with slight blur for realism."""
    uint8 = (arr * 255).astype(np.uint8)
    pil   = Image.fromarray(uint8, mode='L')
    pil   = pil.filter(ImageFilter.GaussianBlur(radius=1.2))
    return pil


def generate_all(output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    base = make_base_chest()

    configs = {
        None:              base,
        "pneumonia":       add_consolidation(base, "right"),
        "pleural_effusion":add_pleural_effusion(base, "left"),
        "cardiomegaly":    add_cardiomegaly(base),
        "nodule":          add_nodule(base),
        "fracture":        add_vertebral_fracture(base),
    }

    for sample in SAMPLES:
        arr  = configs[sample["pathology"]]
        pil  = to_pil(arr)
        path = output_dir / sample["filename"]
        pil.save(path)
        print(f"✓ {sample['filename']:45s} — {sample['label']}")

    print(f"\n{len(SAMPLES)} imagens geradas em: {output_dir.resolve()}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    generate_all(output_dir)
