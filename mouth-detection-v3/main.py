from extract_lips import *



filename = "close.png"
#----------------------------------------------------------------------------
# function used to extract lips points
# input: frame Name + extension
# output: frame, mouth_roi points pair vector
frame, mouth, mouth_roi = extractLips(filename)
