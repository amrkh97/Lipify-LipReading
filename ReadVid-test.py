import os
import cv2
import csv
import glob
from moviepy.video.io.VideoFileClip import VideoFileClip
from dataclasses import dataclass
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



@dataclass()
class WordInstance:
    '''Class for keeping track of each word'''
    name: str
    start_time: float
    end_time: float
    
###################################### Static Data ########################################
FPS = 25
commands = ['bin', 'lay', 'place', 'set']
prepositions = ['at', 'by', 'in', 'with']
colors = ['blue', 'green', 'red', 'white']
adverbs = ['again', 'now', 'please', 'soon']
alphabet = [chr(x) for x in range(ord('a'), ord('z') + 1)] 
numbers = ['one','two','three','four','five','six','seven','eight','nine']
########################################################################################### 
#WIP
def getAlignFileName(Path):
    '''Helper function to get name of the align file for a certain video'''
    return Path.split(".")[0] + ".align"    
    
    
def extractWordTimingFromVideo(filename):
    '''Function that extracts each align file into a list of dataclasses'''
    lines = open(filename).read().splitlines()
    wordsWithTimings = []
    for line in lines:
        temp = line.split()
        wordsWithTimings.append(WordInstance(temp[2], temp[0], temp[1]))
    return wordsWithTimings


def cutVideo(videoPath,fileName,start_time,end_time):
    with VideoFileClip(videoPath) as video:
                new = video.subclip(start_time, end_time)
                new.write_videofile(fileName)

#WIP
def segmentDataSet(Path, Number_Of_Files):
    '''Function responsible for segmenting the whole dataset into separate word files '''
    gen =  glob.iglob(Path+ "*.mpg")
    for i in range(Number_Of_Files):
        py = next(gen)

#WIP
def segmentSingleVideo(videoPath, alignFilePath):
    
    wordTimings = extractWordTimingFromVideo(alignFilePath)
    for word in wordTimings:
        print("Current Word is: {}".format(word.name))
        start_time = round((float(word.start_time)/(FPS*1000)),3)
        end_time = round((float(word.end_time)/(FPS*1000)),3)
        
        if word.name in commands:
            new_index = len(os.listdir('Videos-After-Extraction/Commands/'))
            fileName = 'Videos-After-Extraction/Commands/{}.mp4'.format(new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in prepositions:
            new_index = len(os.listdir('Videos-After-Extraction/Prepositions/'))
            fileName = 'Videos-After-Extraction/Prepositions/{}.mp4'.format(new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in colors:
            new_index = len(os.listdir('Videos-After-Extraction/Colors/'))
            fileName = 'Videos-After-Extraction/Colors/{}.mp4'.format(new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in numbers:
            new_index = len(os.listdir('Videos-After-Extraction/Numbers/'))
            fileName = 'Videos-After-Extraction/Numbers/{}.mp4'.format(new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in adverbs:
            new_index = len(os.listdir('Videos-After-Extraction/Adverb/'))
            fileName = 'Videos-After-Extraction/Adverb/{}.mp4'.format(new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in alphabet:
            new_index = len(os.listdir('Videos-After-Extraction/Alphabet/'))
            fileName = 'Videos-After-Extraction/Alphabet/{}.mp4'.format(new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name == 'sil':
            new_index = len(os.listdir('Videos-After-Extraction/Silence/'))
            fileName = 'Videos-After-Extraction/Silence/{}.mp4'.format(new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
    
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
    pass    

####################################### MAIN CODE ############################################
videoPath = 'DataSet-Trial/bbab8n.mpg'
alignFilePath = 'DataSet-Trial/bbab8n.align'
segmentSingleVideo(videoPath, alignFilePath)
