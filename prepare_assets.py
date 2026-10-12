from pathlib import Path
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

source_path = IMAGES / "source.jpg"
canvas = cv.imread(str(source_path))
if canvas is None:
    raise SystemExit(f"Missing supplied image: {source_path}")
cv.imwrite(str(IMAGES / "scene.png"), canvas)

second = cv.rotate(canvas, cv.ROTATE_180)
second = cv.convertScaleAbs(second, alpha=0.8, beta=35)
cv.imwrite(str(IMAGES / "scene2.png"), second)

print(f"Image assets created in {IMAGES}; sample.avi uses the supplied GIF video")
