from pathlib import Path
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
HSV_LOWER = np.array([15, 45, 70])
HSV_UPPER = np.array([40, 255, 255])
img = cv.imread(str(ROOT / "images" / "scene.png"))
if img is None:
    raise SystemExit("Run prepare_assets.py first")
h, w = img.shape[:2]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)
cv.imwrite(str(OUT / "task4_gray.png"), gray)
cv.imwrite(str(OUT / "task4_hsv.png"), hsv)
cv.imwrite(str(OUT / "task4_color_mask.png"), mask)

cv.imwrite(str(OUT / "task4_resize_small.png"), cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA))
cv.imwrite(str(OUT / "task4_resize_large.png"), cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC))
translation = np.float32([[1, 0, 70], [0, 1, 35]])
cv.imwrite(str(OUT / "task4_translate.png"), cv.warpAffine(img, translation, (w, h)))
rotation = cv.getRotationMatrix2D((w / 2, h / 2), 18, 1.0)
cv.imwrite(str(OUT / "task4_rotate.png"), cv.warpAffine(img, rotation, (w, h)))
src = np.float32([[75, 165], [335, 165], [350, 405], [60, 405]])
dst = np.float32([[0, 0], [300, 0], [300, 180], [0, 180]])
matrix = cv.getPerspectiveTransform(src, dst)
cv.imwrite(str(OUT / "task4_perspective.png"), cv.warpPerspective(img, matrix, (300, 180)))

_, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
_, binary_inv = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
_, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 21, 5)
for name, value in (("binary", binary), ("binary_inv", binary_inv),
                    ("otsu", otsu), ("adaptive", adaptive)):
    cv.imwrite(str(OUT / f"task4_threshold_{name}.png"), value)

filters = {
    "mean": cv.blur(img, (9, 9)),
    "gaussian": cv.GaussianBlur(img, (9, 9), 0),
    "median": cv.medianBlur(img, 9),
    "bilateral": cv.bilateralFilter(img, 9, 75, 75),
}
for name, value in filters.items():
    cv.imwrite(str(OUT / f"task4_smooth_{name}.png"), value)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
morph = {
    "erode": cv.erode(mask, kernel),
    "dilate": cv.dilate(mask, kernel),
    "open": cv.morphologyEx(mask, cv.MORPH_OPEN, kernel),
    "close": cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel),
}
# 本图更适合开运算去噪，因为开运算能去除 mask 中零散的白色前景噪点；闭运算主要用于填补小黑洞。我的刀盾
for name, value in morph.items():
    cv.imwrite(str(OUT / f"task4_morph_{name}.png"), value)
print("Task 4 complete; opening is preferred for isolated foreground noise")
