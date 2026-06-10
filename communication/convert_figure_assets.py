#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert figure bitmap assets in the figures directory to PDF at 1000 dpi.

This is useful for LaTeX workflows where PDF is preferred for figure inclusion.
Note: converting a PNG to PDF does not make it vector art; it only wraps the
bitmap into a PDF container for cleaner manuscript handling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


BITMAP_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PREFERRED_ORDER = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def convert_bitmap_to_pdf(src: Path, dst: Path, dpi: int) -> None:
    with Image.open(src) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            alpha = im.getchannel("A")
            bg.paste(im.convert("RGBA"), mask=alpha)
            out = bg
        else:
            out = im.convert("RGB")
        out.save(dst, "PDF", resolution=float(dpi))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert figure bitmaps to PDF.")
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir).expanduser().resolve()
    converted = 0

    by_stem = {}
    for src in sorted(figures_dir.iterdir()):
        if not src.is_file():
            continue
        suffix = src.suffix.lower()
        if suffix not in BITMAP_SUFFIXES:
            continue
        by_stem.setdefault(src.stem, {})[suffix] = src

    for stem in sorted(by_stem):
        choices = by_stem[stem]
        src = None
        for suffix in PREFERRED_ORDER:
            if suffix in choices:
                src = choices[suffix]
                break
        if src is None:
            continue
        dst = figures_dir / f"{stem}.pdf"
        if dst.exists() and not args.overwrite:
            continue
        convert_bitmap_to_pdf(src, dst, dpi=args.dpi)
        converted += 1
        print(f"[Converted] {src.name} -> {dst.name}")

    print(f"[Done] Converted {converted} figure(s) in {figures_dir}")


if __name__ == "__main__":
    main()
