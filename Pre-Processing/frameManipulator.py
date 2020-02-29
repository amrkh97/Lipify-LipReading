import cv2
import time
from dataclasses import dataclass
from frameHandler import getNumberFramesPerVideo
FPS = 25


@dataclass()
class VideoInfo:
    """Class containing data about each video """
    category: str
    speaker: str
    identifier: str
    completePath: str


def getVideoDataFromPath(videoPath):
    """Function to get data from the video path """
    completeVideoData = videoPath.split('/')[1:]
    currentVideo = VideoInfo(category=completeVideoData[2], speaker=completeVideoData[1],
                             completePath=videoPath, identifier=completeVideoData[3])
    return currentVideo


def getRelativeSilenceVideo(videoPath):
    """Function to get relative silence videos before and after each video"""
    silVid = ['', '']
    vidData = getVideoDataFromPath(videoPath)
    videoNameList = videoPath.split('/')
    tempVidName = videoNameList[0] + '/' + videoNameList[1] + '/' + videoNameList[2] + '/Silence/sil_{}.mp4'
    vidNumber = int((vidData.identifier.split('_')[1]).split('.')[0])
    silVid[0] = tempVidName.format(vidNumber * 2)
    silVid[1] = tempVidName.format((vidNumber * 2) + 1)
    return silVid


def getVideoFrames(videoName):
    """Function to get frames of video in a list"""
    vidcap = cv2.VideoCapture(videoName)
    success, image = vidcap.read()
    allFrames = []  # Get label of video
    while success:
        allFrames.append(image)
        success, image = vidcap.read()
    return allFrames


def isExcess(videoPath):
    """Function that determines if the video has more than specified FPS"""
    if getNumberFramesPerVideo(videoPath) > FPS:
        return True
    else:
        return False


def removeExcessFrames(videoPath):
    pass


def addFramesAtFront(originalVid, silVid1, silVid2):
    newVideo = originalVid
    i = 0
    while len(newVideo) < FPS and i < len(silVid1):
        newVideo.insert(i, silVid1[i])
        i += 1

    while len(newVideo) < FPS and (i - len(silVid1)) < len(silVid2):
        newVideo.insert(i, silVid2[i - len(silVid1)])
        i += 1

    while len(newVideo) < FPS:
        newVideo.insert(0, silVid1[0])

    return newVideo


def addFramesAtEnd(originalVid, silVid1, silVid2):
    newVideo = originalVid
    i = 0
    while len(newVideo) < FPS and i < len(silVid1):
        newVideo.append(silVid1[i])
        i += 1

    while len(newVideo) < FPS and (i - len(silVid1)) < len(silVid2):
        newVideo.append(silVid2[i - len(silVid1)])
        i += 1

    while len(newVideo) < FPS:
        newVideo.insert(0, silVid1[0])

    return newVideo


def addFrames(videoPath, silVid1, silVid2):
    """Add silence frames to video"""
    originalVideo = getVideoFrames(videoPath)
    firstSilVideo = getVideoFrames(silVid1)
    secondSilVideo = getVideoFrames(silVid2)
    # TODO: Create a function to add frames equally at both front and end of video
    addedAtFront = addFramesAtFront(originalVideo, firstSilVideo, secondSilVideo)
    addedAtEnd = addFramesAtEnd(originalVideo, firstSilVideo, secondSilVideo)
    return addedAtFront, addedAtEnd


def manipulateVideo(videoPath):
    """Function that handles action on videos based on the number of frames"""
    # TODO: Find a way to handle excess frames in dataset
    finalVideoList = []
    excessChecker = isExcess(videoPath)
    if excessChecker:
        removeExcessFrames(videoPath)
    else:
        silenceVidNames = getRelativeSilenceVideo(videoPath)
        addFrames(videoPath, silenceVidNames[0], silenceVidNames[1])

    return excessChecker, finalVideoList


if __name__ == "__main__":
    StartTime = time.time()
    videoP = '../Videos-After-Extraction/S1/Adverb/again_3.mp4'
    silVidNames = getRelativeSilenceVideo(videoP)
    L = addFrames(videoP, silVidNames[0], silVidNames[1])
    print("Length Of Videos: {} Frames, {} Frames".format(len(L[0]), len(L[1])))
    print("Run Time: {} seconds.".format(time.time() - StartTime))
