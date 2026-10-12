from pathlib import Path
import argparse
import cv2 as cv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="play frames in a window")
args = parser.parse_args()

cap = cv.VideoCapture(str(ROOT / "images" / "sample.avi"))
if not cap.isOpened():
    raise SystemExit("Cannot open local video")
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv.CAP_PROP_FPS))
# 读取到的 FPS 异常时使用 25.0，保证 VideoWriter 得到有效的正数帧率。奶龙
if not (fps > 0 and fps < 240):
    fps = 25.0
print(f"width={width}, height={height}, fps={fps:.2f}")

writer = cv.VideoWriter(str(OUT / "task2.avi"), cv.VideoWriter_fourcc(*"MJPG"),
                        fps, (width, height))
if not writer.isOpened():
    cap.release()
    raise SystemExit("Cannot create output video")
frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv.putText(frame, f"frame {frame_no}", (15, 35), cv.FONT_HERSHEY_SIMPLEX,
               0.9, (0, 0, 255), 2, cv.LINE_AA)
    writer.write(frame)
    frame_no += 1
    if args.show:
        cv.imshow("video", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
writer.release()
if args.show:
    cv.destroyAllWindows()
print(f"Task 2 complete: {frame_no} frames")
