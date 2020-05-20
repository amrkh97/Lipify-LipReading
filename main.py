from project_prototype import predictOneVideo

if __name__ == "__main__":
    
    predictionFilePath = 'C:/Users/amrkh/Desktop/Projects/Lipify-server/prediction.txt'

    predictionFile = open(predictionFilePath, "w")  # write mode
    predictionFile.write("Hello")
    predictionFile.close()
