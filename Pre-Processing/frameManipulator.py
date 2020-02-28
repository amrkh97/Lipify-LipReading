import cv2
from dataclasses import dataclass


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
    """Function to get frames of video with their label"""
    vidcap = cv2.VideoCapture(videoName)
    success, image = vidcap.read()
    allFrames = [(videoName.split('_')[0]).split('/')[-1]]  # Get label of video
    while success:
        allFrames.append(image)
        success, image = vidcap.read()
    return allFrames


if __name__ == "__main__":
    videoData = '../Videos-After-Extraction/S1/Adverb/again_3.mp4'
    getRelativeSilenceVideo(videoData)
