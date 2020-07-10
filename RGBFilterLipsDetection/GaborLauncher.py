import time
from RGBFilterLipsDetection import GaborSupportFunctions, MouthModel, SkinModel, SkinRegion


def startProcess(img):
    resized = GaborSupportFunctions.resizeImage(img)
    smoothed = GaborSupportFunctions.smoothImg(resized)
    # -----------------------------------------------
    skin = SkinModel.getSkin(smoothed)
    region, status = SkinRegion.extractSkinRegions(smoothed, skin)
    if not status:
        frame = GaborSupportFunctions.resizeImage(resized, (150, 100))
        return frame, False
    else:
        mouthROI = MouthModel.extractMouthROI(resized, region[0], region[1], region[2], region[3])
        mouthROI = GaborSupportFunctions.resizeImage(mouthROI, (150, 100))
        return mouthROI, True


# -----------------------------------------------
if __name__ == "__main__":
    startTime = time.time()
    videoPath = "../Prototype-Test-Videos/Colors_2.mp4"
    frames, status = GaborSupportFunctions.getVideoFrames(videoPath)
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
    else:
        print("Failed To Get Video")
    print(time.time() - startTime)
