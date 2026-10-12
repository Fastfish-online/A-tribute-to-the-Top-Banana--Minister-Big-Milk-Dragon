from pathlib import Path
import argparse
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="open GUI windows")
args = parser.parse_args()

path = ROOT / "images" / "scene.png"
color = cv.imread(str(path), cv.IMREAD_COLOR)
gray = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
if color is None or gray is None:
    raise SystemExit(f"Image path is invalid: {path}")
cv.imwrite(str(OUT / "task1_gray.png"), gray)

board = np.zeros((480, 640, 3), np.uint8)
cv.line(board, (30, 40), (280, 40), (0, 255, 255), 4, cv.LINE_AA)
cv.rectangle(board, (30, 80), (240, 210), (255, 120, 0), 3)
cv.circle(board, (410, 145), 65, (0, 255, 0), 3, cv.LINE_AA)
cv.ellipse(board, (520, 330), (85, 45), 25, 0, 300, (255, 0, 255), 3)
polygon = np.array([[45, 420], [145, 260], [270, 410], [180, 450]], np.int32)
cv.polylines(board, [polygon], True, (0, 0, 255), 3, cv.LINE_AA)
cv.putText(board, "OpenCV drawing", (300, 260), cv.FONT_HERSHEY_SIMPLEX,
           0.9, (255, 255, 255), 2, cv.LINE_AA)
cv.imwrite(str(OUT / "task1_canvas.png"), board)

annotated = color.copy()
cv.rectangle(annotated, (35, 5), (375, 535), (0, 255, 255), 3)
cv.putText(annotated, "character 0.80", (35, 32), cv.FONT_HERSHEY_SIMPLEX,
           0.75, (0, 0, 0), 2, cv.LINE_AA)
cv.imwrite(str(OUT / "task1_annotated.png"), annotated)

if args.show:
    cv.imshow("color", color)
    cv.imshow("gray", gray)
    cv.imshow("drawing", board)
    cv.waitKey(0)
    cv.destroyAllWindows()
print("Task 1 complete")
