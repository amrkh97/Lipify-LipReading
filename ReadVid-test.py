from imutils import face_utils
import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib-predictor.dat")

# load the input image, resize it, and convert it to grayscale
image = cv2.imread("test.jpg")

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(image, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):

	shape = predictor(image, rect)
	shape = face_utils.shape_to_np(shape)
 
	for (x, y) in shape[49:]:
		cv2.circle(image, (x, y), 2, (0, 0, 0))

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
import cv2
def getVideoFrames(category, videoName):
    vidcap = cv2.VideoCapture('Videos-After-Extraction/{}/{}.mp4'.format(category,videoName))
    success,image = vidcap.read()
    allFrames = []
    allFrames.append(videoName.split('_')[0]) #Label
    while success:
        allFrames.append(image)
        success,image = vidcap.read()
    return allFrames   

#print(getVideoFrames('Alphabet','c_10596')[0])
'''

