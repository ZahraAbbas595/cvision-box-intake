"""Create a labelled contact sheet for manual evaluation-set review."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "eval" / "dataset" / "images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "results" / "contact_sheet.jpg",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    label_height = 28
    rows = math.ceil(len(image_paths) / args.columns)
    sheet = Image.new(
        "RGB",
        (
            args.columns * args.thumb_width,
            rows * (args.thumb_height + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)

    for index, image_path in enumerate(image_paths):
        column = index % args.columns
        row = index // args.columns
        x = column * args.thumb_width
        y = row * (args.thumb_height + label_height)
        with Image.open(image_path) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            thumbnail = ImageOps.contain(
                normalized,
                (args.thumb_width, args.thumb_height),
                Image.Resampling.LANCZOS,
            )
        image_x = x + (args.thumb_width - thumbnail.width) // 2
        image_y = y + (args.thumb_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (image_x, image_y))
        draw.text(
            (x + 8, y + args.thumb_height + 4),
            image_path.name,
            fill="black",
            font=font,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
