# /// script
# requires-python = ">=3.12"
# dependencies = [
# "pillow"
# ]
# ///

from pathlib import Path
import sys
from PIL import Image
import argparse

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297


def parse_args() -> tuple[str, float, int]:
    parser = argparse.ArgumentParser(description="""Saquare A4 image printer.""")
    parser.add_argument("-i", "--image", type=str, required=True)
    parser.add_argument(
        "-s", "--size", type=float, required=True, help="Output size in mm."
    )
    parser.add_argument("-p", "--ppi", type=int, default=96, help="Pixels per inch.")
    args = parser.parse_args()
    return args.image, args.size, args.ppi


def main() -> None:
    iamge_path, image_size_mm, ppi = parse_args()
    image_path = Path(iamge_path)
    if not image_path.exists():
        print(f"Image {image_path} doesn't exist!")
        sys.exit(1)
    img: Image.Image = Image.open(image_path).convert("RGB")
    ppm = ppi / 25.4  ## pixels per mm
    target_width = int(ppm * image_size_mm)
    height, width = img.height, img.width
    if height != width:
        img = img.crop((0, 0, target_width, target_width))
        print("WARNING, cropping")

    img = img.resize(target_width, target_width)

    a4_img_width = int(A4_WIDTH_MM * ppm)
    a4_img_height = int(A4_HEIGHT_MM * ppm)
    out = Image.new("RGB", (a4_img_width, a4_img_height))
    for h0 in range(0, a4_img_height, target_width):
        for w0 in range(0, a4_img_width, target_width):
            out.paste(img, (w0, h0))

    out_name = "out.png"
    print(f"Saving file to {out_name}")
    out.save(out_name)
    sys.exit(0)


if __name__ == "__main__":
    main()
