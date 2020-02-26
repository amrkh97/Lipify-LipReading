import glob
import os
from dataclasses import dataclass
from moviepy.video.io.VideoFileClip import VideoFileClip


@dataclass()
class WordInstance:
    """Class for keeping track of each word """
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
numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
########################################################################################### 


def getAlignFileName(Path):
    """Helper function to get name of the align file for a certain video """
    tempPath = Path.split("/")
    newPath = tempPath[0] + '/align/' + (tempPath[2].split('\\')[1]).split('.')[0] + '.align' 
    return newPath


def extractWordTimingFromVideo(filename):
    """Function that extracts each align file into a list of dataclasses """
    lines = open(filename).read().splitlines()
    wordsWithTimings = []
    for line in lines:
        temp = line.split()
        wordsWithTimings.append(WordInstance(temp[2], temp[0], temp[1]))
    return wordsWithTimings


def cutVideo(videoPath, fileName, start_time, end_time):
    """Function that cuts video from a given interval """
    with VideoFileClip(videoPath, audio=False) as video:
        new = video.subclip(start_time, end_time)
        new.write_videofile(fileName)
        new.close()


def createDataSetDirectories(speakerNumber):
    """Function to create dataset segmentation directories for each individual speaker"""
    dirName = 'Videos-After-Extraction/S{}/Commands/'.format(speakerNumber)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    dirName = 'Videos-After-Extraction/S{}/Prepositions/'.format(speakerNumber)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    dirName = 'Videos-After-Extraction/S{}/Colors/'.format(speakerNumber)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    dirName = 'Videos-After-Extraction/S{}/Adverb/'.format(speakerNumber)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    dirName = 'Videos-After-Extraction/S{}/Alphabet/'.format(speakerNumber)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    dirName = 'Videos-After-Extraction/S{}/Numbers/'.format(speakerNumber)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    dirName = 'Videos-After-Extraction/S{}/Silence/'.format(speakerNumber)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    

def segmentSingleVideo(videoPath, alignFilePath, speakerNumber):
    """Segment a single video into its underlying words in their prespective folders """   
    wordTimings = extractWordTimingFromVideo(alignFilePath)
    
    createDataSetDirectories(speakerNumber)
    
    for word in wordTimings:
       
        start_time = round((float(word.start_time)/(FPS*1000)), 3)
        end_time = round((float(word.end_time)/(FPS*1000)), 3)
        
        if word.name in commands:
            new_index = len(os.listdir('Videos-After-Extraction/S{}/Commands/'.format(speakerNumber)))
            fileName = 'Videos-After-Extraction/S{}/Commands/{}_{}.mp4'.format(speakerNumber, word.name, new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in prepositions:
            new_index = len(os.listdir('Videos-After-Extraction/S{}/Prepositions/'.format(speakerNumber)))
            fileName = 'Videos-After-Extraction/S{}/Prepositions/{}_{}.mp4'.format(speakerNumber, word.name, new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in colors:
            new_index = len(os.listdir('Videos-After-Extraction/S{}/Colors/'.format(speakerNumber)))
            fileName = 'Videos-After-Extraction/S{}/Colors/{}_{}.mp4'.format(speakerNumber, word.name, new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in numbers:
            new_index = len(os.listdir('Videos-After-Extraction/S{}/Numbers/'.format(speakerNumber)))
            fileName = 'Videos-After-Extraction/S{}/Numbers/{}_{}.mp4'.format(speakerNumber, word.name, new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in adverbs:
            new_index = len(os.listdir('Videos-After-Extraction/S{}/Adverb/'.format(speakerNumber)))
            fileName = 'Videos-After-Extraction/S{}/Adverb/{}_{}.mp4'.format(speakerNumber, word.name, new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name in alphabet:
            new_index = len(os.listdir('Videos-After-Extraction/S{}/Alphabet/'.format(speakerNumber)))
            fileName = 'Videos-After-Extraction/S{}/Alphabet/{}_{}.mp4'.format(speakerNumber, word.name, new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
        
        if word.name == 'sil':
            new_index = len(os.listdir('Videos-After-Extraction/S{}/Silence/'.format(speakerNumber)))
            fileName = 'Videos-After-Extraction/S{}/Silence/{}_{}.mp4'.format(speakerNumber, word.name, new_index)
            cutVideo(videoPath, fileName, start_time, end_time)
    

def segmentDataSet(Path, Number_Of_Speakers):
    """Function responsible for segmenting the whole dataset into separate word files """
    for i in range(Number_Of_Speakers):
        videoPath = Path + "video/S{}/".format(i+1) + "*.mpg"
        videosGen = glob.iglob(videoPath)
        try:
            for j in range(len(os.listdir(videoPath.split('*')[0]))):
                py = next(videosGen)
                segmentSingleVideo(py, getAlignFileName(py), i+1)
        except StopIteration:
            print("Segmented the dataset.") 


####################################### MAIN CODE ############################################
dataSetPath = 'GP DataSet/'
numberSpeakers = 20
segmentDataSet(dataSetPath, numberSpeakers)

