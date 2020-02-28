from imutils import face_utils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib-predictor.dat")
image = cv2.imread("test.jpg")
rects = detector(image, 1)

for (i, rect) in enumerate(rects):
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)
    mouthPoints = shape[49:]
    for (x, y) in shape[49:]:  # Mouth points only
        cv2.circle(image, (x, y), 2, (0, 0, 0))

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
