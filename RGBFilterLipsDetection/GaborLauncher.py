import time

from GaborSupportFunctions import resizeImage, smoothImg, getVideoFrames
from MouthModel import extractMouthROI
from SkinModel import getSkin
from SkinRegion import extractSkinRegions


def startProcess(img):
    # -----------------------------------------------
    # img = readFrame()
    # -----------------------------------------------
    resized = resizeImage(img)
    # -----------------------------------------------
    smoothed = smoothImg(resized)
    # -----------------------------------------------
    skin = getSkin(smoothed)
    region, status = extractSkinRegions(smoothed, skin)
    if not status:
        frame = resizeImage(resized, (150, 100))
        return frame, False
    else:
        mouthROI = extractMouthROI(resized, region[0], region[1], region[2], region[3])
        mouthROI = resizeImage(mouthROI, (150, 100))
        return mouthROI, True


# -----------------------------------------------
if __name__ == "__main__":
    startTime = time.time()
    videoPath = "../Prototype-Test-Videos/Colors_2.mp4"
    frames, status = getVideoFrames(videoPath)
    if status:
        detected = []
        for i, frame in enumerate(frames):
            print("Frame {}".format(i + 1))
            lips, status = startProcess(frame)
            if not status:
                print("failed to get face")
                detected.append(frame)
            else:
                detected.append(lips)
            # cv2.imshow(str(i), detected[-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Failed To Get Video")
    print(time.time() - startTime)
