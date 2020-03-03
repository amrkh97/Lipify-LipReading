import glob
import os
import random
import time
from dataclasses import dataclass

import cv2

from frameHandler import getNumberFramesPerVideo

FPS = 30
FrameSize = (360, 288)
random.seed(time.time())


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
    allFrames = []
    while success:
        allFrames.append(image)
        success, image = vidcap.read()
    return allFrames


def isExcess(videoPath):
    """Function that determines if the video has more than specified FPS"""
    if getNumberFramesPerVideo(videoPath) > FPS:
        return True
    return False


def removeExcessFrames(videoPath):
    """Function to remove frames randomly from a video"""
    originalVideo = getVideoFrames(videoPath)
    originalVideo.pop(0)  # Remove first frame by default

    if len(originalVideo) > FPS:
        originalVideo.pop()  # Remove last frame by default

    while len(originalVideo) > FPS:
        popIndex = random.randint(0, len(originalVideo))
        originalVideo.pop(popIndex)

    return originalVideo


def addFramesAtFront(originalVid, silVid1, silVid2):
    """Function to add silence frames at the front of the video"""
    newVideo = originalVid
    i = 0
    while len(newVideo) < FPS and i < len(silVid1):
        newVideo.insert(i, silVid1[i])
        i += 1

    while len(newVideo) < FPS and (i - len(silVid1)) < len(silVid2):
        newVideo.insert(i, silVid2[i - len(silVid1)])
        i += 1

    i = 0
    while len(newVideo) < FPS:
        newVideo.insert(0, silVid1[i])
        i += 1
        i = i % len(silVid1)

    return newVideo


def addFramesAtEnd(originalVid, silVid1, silVid2):
    """Function to add silence frames at the end of the video"""
    newVideo = originalVid
    i = 0
    while len(newVideo) < FPS and i < len(silVid1):
        newVideo.append(silVid1[i])
        i += 1

    while len(newVideo) < FPS and (i - len(silVid1)) < len(silVid2):
        newVideo.append(silVid2[i - len(silVid1)])
        i += 1

    i = 0
    while len(newVideo) < FPS:
        newVideo.append(silVid1[i])
        i += 1
        i = i % len(silVid1)

    return newVideo


def addFramesAtFrontAndEnd(originalVid, silVid1, silVid2):
    """Function that addes silence frames at both front and end of a video"""
    if FPS >= (len(originalVid) + len(silVid1) + len(silVid2)):
        newVideo = silVid1 + originalVid + silVid2
        newVideo = addFramesAtFront(newVideo, silVid1, silVid2)
        return newVideo
    elif FPS > (len(silVid1) + len(originalVid)):
        newVideo = silVid1 + originalVid
        newVideo = addFramesAtEnd(newVideo, silVid2, silVid2)
        return newVideo
    elif FPS > (len(silVid2) + len(originalVid)):
        newVideo = originalVid + silVid2
        newVideo = addFramesAtFront(newVideo, silVid1, silVid1)
        return newVideo

    newVideo = addFramesAtEnd(originalVid, silVid1, silVid2)  # Default case
    return newVideo


def addFrames(videoPath, silVid1, silVid2):
    """Add silence frames to video"""
    originalVideo = getVideoFrames(videoPath)
    firstSilVideo = getVideoFrames(silVid1)
    secondSilVideo = getVideoFrames(silVid2)

    addedAtFront = addFramesAtFront(originalVideo, firstSilVideo, secondSilVideo)
    addedAtEnd = addFramesAtEnd(originalVideo, firstSilVideo, secondSilVideo)
    addedAtFrontAndEnd = addFramesAtFrontAndEnd(originalVideo, firstSilVideo, secondSilVideo)

    return addedAtFront, addedAtEnd, addedAtFrontAndEnd


def manipulateVideo(videoPath):
    """Function that handles action on videos based on the number of frames"""
    finalVideoList = []
    if isExcess(videoPath):
        finalVideoList = [removeExcessFrames(videoPath), ]
    else:
        silenceVidNames = getRelativeSilenceVideo(videoPath)
        finalVideoList = addFrames(videoPath, silenceVidNames[0], silenceVidNames[1])

    return finalVideoList


def createDataSetDirectories(speakerNumber):
    """Function to create new dataset segmentation directories for each individual speaker"""
    dirPath = '../New-DataSet-Videos/S{}'.format(speakerNumber)

    dirName = dirPath + '/Commands/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    dirName = dirPath + '/Prepositions/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    dirName = dirPath + '/Colors/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    dirName = dirPath + '/Adverb/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    dirName = dirPath + '/Alphabet/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    dirName = dirPath + '/Numbers/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)


def saveVideoToPath(videoData, videoFrames):
    """Function to save frames into a video"""
    saveDirectory = '../New-DataSet-Videos/{}/{}/'.format(videoData.speaker, videoData.category)
    fileIndex = len(os.listdir(saveDirectory))
    fileIdentifier = videoData.identifier.split('_')[0]
    fileName = '{}_{}.mp4'.format(fileIdentifier, fileIndex)
    outVideo = cv2.VideoWriter(saveDirectory + fileName, cv2.VideoWriter_fourcc(*'mp4v'), FPS, FrameSize)

    for frame in videoFrames:
        outVideo.write(frame)
    outVideo.release()


def adjustDataSetTo25Frames(Number_Of_Speakers):
    """Manipulates the whole dataset videos to be 30 frames"""

    categories = ['Adverb', 'Alphabet', 'Commands', 'Colors', 'Numbers', 'Prepositions']
    for i in range(Number_Of_Speakers):
        for category in categories:
            videoPath = "../Videos-After-Extraction/S{}/{}/".format(i + 1, category) + "*.mp4"
            vidList = glob.glob(videoPath)

            def f(x):
                return x.replace("\\", '/')

            vidList = [f(x) for x in vidList]

            createDataSetDirectories(i + 1)

            try:
                for idx, j in enumerate(vidList):
                    vidData = getVideoDataFromPath(j)
                    adjustedVideoList = manipulateVideo(j)
                    _ = [saveVideoToPath(vidData, x) for x in adjustedVideoList]
            except StopIteration:
                print("Segmented the new dataset.")
            print("Finished Segmenting Category: {}".format(category))

        print("Finished Speaker Number: {}".format(i + 1))
        print("=========================================")


if __name__ == "__main__":
    StartTime = time.time()
    numberOfSpeakers = 20
    adjustDataSetTo25Frames(numberOfSpeakers)
    print("Run Time: {} seconds.".format(time.time() - StartTime))
