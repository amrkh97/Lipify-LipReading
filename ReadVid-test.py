import cv2

    
#Test function for videos to check that frames are correct    
def getVideoFrames():
    vidcap = cv2.VideoCapture('Videos-After-Extraction/Adverb/0.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imshow("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(count)   