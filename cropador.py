import cv2
import numpy as np

polygon_points = []

def click_event(event, x, y, flags, param):
    global polygon_points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        if len(polygon_points) > 1:
            cv2.line(image, polygon_points[-2], polygon_points[-1], (255, 0, 0), 2)

        cv2.imshow("imagem", image)

image = cv2.imread("datasets/imagens/esalq.png")
cv2.namedWindow("imagem")
cv2.setMouseCallback("imagem", click_event)

cv2.imshow("imagem", image)
cv2.waitKey(0)

if len(polygon_points) > 2:
    mask = np.zeros_like(image, dtype=np.uint8)
    pts = np.array(polygon_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    result = cv2.bitwise_and(image, mask)

    cv2.imshow("Imagem cropada", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.savetxt("crop_points.txt", polygon_points)
else:
    print("Pelo menos 3 pontos para um poligono")
    cv2.destroyAllWindows()
