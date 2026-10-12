from pathlib import Path
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
img = cv.imread(str(ROOT / "images" / "scene.png"))
img2 = cv.imread(str(ROOT / "images" / "scene2.png"))
if img is None or img2 is None:
    raise SystemExit("Run prepare_assets.py first")
print("shape:", img.shape, "size:", img.size, "dtype:", img.dtype)
y, x = 100, 100
print("pixel before:", img[y, x].tolist())
img[y, x] = (0, 255, 255)
print("pixel after:", img[y, x].tolist())

roi_result = img.copy()
height, width = roi_result.shape[:2]
src = roi_result[40:140, 40:140]
target = roi_result[height-120:height-20, width-120:width-20]
if src.shape != target.shape:
    raise ValueError("ROI source and target sizes do not match")
target[:] = src
cv.imwrite(str(OUT / "task3_roi.png"), roi_result)

b, g, r = cv.split(img)
for name, channel in zip("bgr", (b, g, r)):
    cv.imwrite(str(OUT / f"task3_channel_{name}.png"), channel)
r_brighter = cv.add(r, 70)
cv.imwrite(str(OUT / "task3_red_brighter.png"), cv.merge((b, g, r_brighter)))

cv.imwrite(str(OUT / "task3_border_constant.png"),
           cv.copyMakeBorder(img, 30, 30, 30, 30, cv.BORDER_CONSTANT, value=(0, 255, 255)))
cv.imwrite(str(OUT / "task3_border_reflect.png"),
           cv.copyMakeBorder(img, 30, 30, 30, 30, cv.BORDER_REFLECT_101))

img2 = cv.resize(img2, (img.shape[1], img.shape[0]))
# NumPy 的 uint8 “+” 对超过 255 的值按模 256 回绕；cv.add 会把超过 255 的值饱和到 255。Milkdragon
cv.imwrite(str(OUT / "task3_numpy_add.png"), img + img2)
cv.imwrite(str(OUT / "task3_cv_add.png"), cv.add(img, img2))
cv.imwrite(str(OUT / "task3_blend.png"), cv.addWeighted(img, 0.6, img2, 0.4, 0))

mask = np.zeros(img.shape[:2], np.uint8)
cv.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), min(img.shape[:2]) // 3, 255, -1)
cv.imwrite(str(OUT / "task3_mask_circle.png"), cv.bitwise_and(img, img, mask=mask))

small = cv.resize(img, (160, 120))
start = cv.getTickCount()
manual = np.empty(small.shape[:2], np.uint8)
for row in range(small.shape[0]):
    for col in range(small.shape[1]):
        blue, green, red = map(int, small[row, col])
        manual[row, col] = round(0.114 * blue + 0.587 * green + 0.299 * red)
manual_ms = (cv.getTickCount() - start) * 1000 / cv.getTickFrequency()
start = cv.getTickCount()
fast = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
fast_ms = (cv.getTickCount() - start) * 1000 / cv.getTickFrequency()
print(f"manual loop: {manual_ms:.3f} ms; cv.cvtColor: {fast_ms:.3f} ms")
print("Task 3 complete")
