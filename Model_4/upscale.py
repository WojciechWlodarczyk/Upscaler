#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
from tqdm import tqdm

# (opcjonalne) delikatne odszumianie, jeśli masz opencv-python
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def find_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        for ext in SUPPORTED_EXTS:
            yield from path.rglob(f"*{ext}")


def load_image_keep_exif(p: Path):
    img = Image.open(p)
    exif = img.info.get("exif", None)
    return img, exif


def denoise_if_requested(img: Image.Image, strength: int) -> Image.Image:
    """
    Proste odszumianie (bilateral filter) przez OpenCV bez zmiany rozmiaru.
    strength: 0-100 (0 = wyłączone)
    """
    if strength <= 0 or not HAS_CV2:
        return img

    # konwersja PIL -> OpenCV (BGR)
    rgb = img.convert("RGB")
    import numpy as np  # lokalny import by nie wymagać jeśli nie trzeba
    arr = np.array(rgb)[:, :, ::-1]
    # Parametry filtra: dobry kompromis dla zdjęć
    d = 9
    sigmaColor = 25 + int(2 * strength)
    sigmaSpace = 25 + int(2 * strength)
    den = cv2.bilateralFilter(arr, d, sigmaColor, sigmaSpace)
    # powrót do PIL
    den_rgb = den[:, :, ::-1]
    return Image.fromarray(den_rgb)


def upscale(img: Image.Image, scale: float, method: str) -> Image.Image:
    if scale <= 0:
        raise ValueError("Skala musi być > 0")

    methods = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST,
    }
    resample = methods.get(method.lower())
    if resample is None:
        raise ValueError(f"Nieznana metoda '{method}'. Dostępne: {', '.join(methods)}")

    w, h = img.size
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    # zachowanie profilu kolorów/alpha
    img = ImageOps.exif_transpose(img)  # uwzględnij orientację EXIF
    return img.resize(new_size, resample=resample)


def save_image(img: Image.Image, out_path: Path, exif, quality: int, webp_quality: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    save_kwargs = {}

    if exif and ext in {".jpg", ".jpeg"}:
        save_kwargs["exif"] = exif

    if ext in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = "4:4:4"
        save_kwargs["optimize"] = True
    elif ext == ".webp":
        save_kwargs["quality"] = webp_quality
        save_kwargs["method"] = 6
    elif ext in {".png"}:
        save_kwargs["optimize"] = True

    img.save(out_path.as_posix(), **save_kwargs)


def main():
    ap = argparse.ArgumentParser(
        description="Prosty upscaler obrazów (Pillow). Obsługuje plik lub folder."
    )
    ap.add_argument("input", type=str, help="Plik wejściowy lub folder")
    ap.add_argument("-o", "--output", type=str, default="upscaled",
                    help="Folder wyjściowy (domyślnie: ./upscaled)")
    ap.add_argument("-s", "--scale", type=float, default=2.0,
                    help="Współczynnik skalowania, np. 2.0 (x2), 4.0 (x4)")
    ap.add_argument("-m", "--method", type=str, default="lanczos",
                    choices=["lanczos", "bicubic", "bilinear", "nearest"],
                    help="Metoda resamplingu (domyślnie: lanczos)")
    ap.add_argument("--suffix", type=str, default="_x{scale}",
                    help="Sufiks do nazwy pliku, np. '_x{scale}' -> _x2.0")
    ap.add_argument("--jpg-quality", type=int, default=95,
                    help="Jakość zapisu JPG (1-100, domyślnie 95)")
    ap.add_argument("--webp-quality", type=int, default=95,
                    help="Jakość zapisu WebP (1-100, domyślnie 95)")
    ap.add_argument("--denoise", type=int, default=0,
                    help="Odszumianie 0-100 (wymaga opencv-python). 0 = wyłączone.")
    ap.add_argument("--keep-format", action="store_true",
                    help="Zachowaj oryginalne rozszerzenie (domyślnie zachowuje).")
    ap.add_argument("--to-format", type=str, default="",
                    help="Wymuś format wyjściowy: jpg/png/webp/tiff/bmp")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)

    if not in_path.exists():
        print(f"Nie znaleziono: {in_path}", file=sys.stderr)
        sys.exit(1)

    imgs = [p for p in find_images(in_path) if p.suffix.lower() in SUPPORTED_EXTS]
    if not imgs:
        print("Brak obrazów do przetworzenia.", file=sys.stderr)
        sys.exit(1)

    if args.denoise > 0 and not HAS_CV2:
        print("[INFO] --denoise ustawione, ale OpenCV nie jest dostępne. Pomijam odszumianie.", file=sys.stderr)

    for p in tqdm(imgs, desc="Upscaling", unit="img"):
        try:
            img, exif = load_image_keep_exif(p)
            if args.denoise > 0 and HAS_CV2:
                img = denoise_if_requested(img, args.denoise)
            up = upscale(img, args.scale, args.method)

            # ustalenie rozszerzenia wyjściowego
            if args.to_format:
                ext_map = {
                    "jpg": ".jpg", "jpeg": ".jpg", "png": ".png",
                    "webp": ".webp", "tiff": ".tiff", "tif": ".tiff", "bmp": ".bmp"
                }
                out_ext = ext_map.get(args.to_format.lower())
                if not out_ext:
                    raise ValueError("Nieprawidłowy --to-format (dozw.: jpg/png/webp/tiff/bmp)")
            else:
                out_ext = p.suffix  # zachowaj format

            suffix = args.suffix.format(scale=args.scale)
            out_name = f"{p.stem}{suffix}{out_ext}"
            out_path = out_dir / p.relative_to(in_path if in_path.is_dir() else p.parent)
            out_path = out_path.parent / out_name

            save_image(up, out_path, exif, args.jpg-quality, args.webp_quality)

        except Exception as e:
            print(f"[BŁĄD] {p}: {e}", file=sys.stderr)

    print(f"Gotowe. Wyniki w: {out_dir.resolve()}")


# if __name__ == "__main__":
#    main()
"""
upscale(Image.open('screenshot_1.png'), 2, 'lanczos').save('scPo_1lanczos.png')
upscale(Image.open('screenshot_1.png'), 2, 'bicubic').save('scPo_1bicubic.png')
upscale(Image.open('screenshot_1.png'), 2, 'bilinear').save('scPo_1bilinear.png')
upscale(Image.open('screenshot_1.png'), 2, 'nearest').save('scPo_1nearest.png')
"""

