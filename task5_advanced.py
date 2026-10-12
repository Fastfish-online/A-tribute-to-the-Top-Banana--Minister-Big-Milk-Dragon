from pathlib import Path
import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

MIN_CONTOUR_AREA = 500
CANNY_1 = (40, 100)
CANNY_2 = (100, 200)

image = cv.imread(str(ROOT / "images" / "source.jpg"), cv.IMREAD_COLOR)
if image is None:
    raise SystemExit("读取 images/source.jpg 失败，请检查图片路径。")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def gradients():
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    cv.imwrite(str(OUT / "task5_sobel_x.png"), cv.convertScaleAbs(sobel_x))
    cv.imwrite(str(OUT / "task5_sobel_y.png"), cv.convertScaleAbs(sobel_y))
    cv.imwrite(str(OUT / "task5_laplacian.png"), cv.convertScaleAbs(laplacian))


def canny_edges():
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges_1 = cv.Canny(blurred, *CANNY_1)
    edges_2 = cv.Canny(blurred, *CANNY_2)
    cv.imwrite(str(OUT / "task5_canny_40_100.png"), edges_1)
    cv.imwrite(str(OUT / "task5_canny_100_200.png"), edges_2)
    # 100/200 会过滤更多弱边缘，因此比 40/100 的结果更干净。我像只鱼儿在你的荷塘
    print("Canny 对比：100/200 的弱边缘更少，结果更干净。")
    return edges_1


def pyramid():
    down_1 = cv.pyrDown(image)
    down_2 = cv.pyrDown(down_1)
    up_1 = cv.pyrUp(down_2)
    cv.imwrite(str(OUT / "task5_pyr_down_1.png"), down_1)
    cv.imwrite(str(OUT / "task5_pyr_down_2.png"), down_2)
    cv.imwrite(str(OUT / "task5_pyr_up.png"), up_1)
    # 两次降采样后再放大，图像仍比原图小，边缘更模糊，已经丢失的细节不能恢复。有一个（整人的）好看的魔法网站推荐，在readme结尾，可以看看


def contours():
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    binary = cv.inRange(hsv, np.array([10, 35, 70]), np.array([40, 255, 255]))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
    cv.imwrite(str(OUT / "task5_contour_binary.png"), binary)

    found, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not found:
        raise RuntimeError("没有找到轮廓，请检查二值化结果。")
    large = [contour for contour in found if cv.contourArea(contour) > MIN_CONTOUR_AREA]
    biggest = max(found, key=cv.contourArea)
    biggest_area = cv.contourArea(biggest)

    result = image.copy()
    cv.drawContours(result, found, -1, (128, 128, 128), 1, cv.LINE_AA)
    cv.drawContours(result, large, -1, (0, 255, 255), 2, cv.LINE_AA)

    x, y, width, height = cv.boundingRect(biggest)
    cv.rectangle(result, (x, y), (x + width, y + height), (0, 0, 255), 2, cv.LINE_AA)
    rotated_box = cv.boxPoints(cv.minAreaRect(biggest)).astype(np.int32)
    cv.polylines(result, [rotated_box], True, (255, 0, 0), 2, cv.LINE_AA)

    label = f"area={biggest_area:.0f} px^2"
    text_y = max(24, y - 10)
    cv.putText(result, label, (max(5, x), text_y), cv.FONT_HERSHEY_SIMPLEX,
               0.62, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(result, label, (max(5, x), text_y), cv.FONT_HERSHEY_SIMPLEX,
               0.62, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(result, f"threshold>{MIN_CONTOUR_AREA}", (8, image.shape[0] - 12),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
    cv.imwrite(str(OUT / "task5_contours.png"), result)
    print(f"轮廓总数={len(found)}，面积>{MIN_CONTOUR_AREA} 的轮廓数={len(large)}，最大面积={biggest_area:.0f}")


def histogram():
    equalized = cv.equalizeHist(gray)
    cv.imwrite(str(OUT / "task5_hist_before.png"), gray)
    cv.imwrite(str(OUT / "task5_hist_equalized.png"), equalized)
    histogram_data = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    level = int(np.argmax(histogram_data))
    print(f"数量最多的灰度级={level}，像素个数={int(histogram_data[level])}")


def template_matching():
    x1, y1, x2, y2 = 145, 5, 275, 125
    template = image[y1:y2, x1:x2].copy()
    response = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv.minMaxLoc(response)
    height, width = template.shape[:2]
    result = image.copy()
    cv.rectangle(result, top_left, (top_left[0] + width, top_left[1] + height),
                 (0, 0, 255), 3, cv.LINE_AA)
    cv.putText(result, f"score={score:.3f}", (top_left[0], min(image.shape[0] - 8, top_left[1] + height + 22)),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv.LINE_AA)
    cv.imwrite(str(OUT / "task5_template.png"), template)
    cv.imwrite(str(OUT / "task5_template_match.png"), result)


def hough_lines():
    source = np.zeros((420, 620, 3), np.uint8)
    for p1, p2 in [((40, 65), (580, 65)), ((65, 130), (550, 350)), ((90, 380), (535, 190))]:
        cv.line(source, p1, p2, (255, 255, 255), 5, cv.LINE_AA)
    edges = cv.Canny(cv.cvtColor(source, cv.COLOR_BGR2GRAY), 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=80, maxLineGap=15)
    result = source.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2, cv.LINE_AA)
    cv.imwrite(str(OUT / "task5_hough_lines.png"), result)


def hough_circles():
    source = np.zeros((440, 640, 3), np.uint8)
    expected = [((150, 150), 65), ((360, 140), 90), ((500, 315), 55)]
    for center, radius in expected:
        cv.circle(source, center, radius, (255, 255, 255), -1, cv.LINE_AA)
    gray_circles = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    gray_circles = cv.GaussianBlur(gray_circles, (9, 9), 2)
    detected = cv.HoughCircles(gray_circles, cv.HOUGH_GRADIENT, 1.2, 100,
                               param1=120, param2=30, minRadius=35, maxRadius=110)
    result = source.copy()
    if detected is not None:
        for x, y, radius in np.uint16(np.around(detected)).reshape(-1, 3):
            cv.circle(result, (x, y), radius, (0, 0, 255), 3, cv.LINE_AA)
            cv.circle(result, (x, y), 3, (0, 255, 0), -1, cv.LINE_AA)
            cv.putText(result, f"r={radius}", (int(x) + 5, int(y) - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.imwrite(str(OUT / "task5_hough_circles.png"), result)


def main():
    gradients()
    canny_edges()
    pyramid()
    contours()
    histogram()
    template_matching()
    hough_lines()
    hough_circles()
    print("任务 5 完成。")


if __name__ == "__main__":
    main()
