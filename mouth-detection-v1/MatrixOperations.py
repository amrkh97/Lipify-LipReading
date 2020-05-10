# -----------------------------------------------------------------------------------------
# function that Create a default value matrix
# input: length, width, default value
# output: matrix of default values
def createMatrix(rowCount, colCount, data):
    mat = []
    for i in range(rowCount):
        rowList = []
        for j in range(colCount):
            rowList.append(data)
        mat.append(rowList)
    return mat
# -----------------------------------------------------------------------------------------
# function that makes a logical AND List operation
# input: list 1, list 2
# output: list of True/False status
def logic_and_list(lis1,lis2):
    lis = []
    for i in range(0,len(lis1)):
        rowlist = []
        for j in range(0,len(lis1[i])):
            if lis1[i][j] == True and lis2[i][j] == True:
                rowlist.append(True)
            else:
                rowlist.append(False)
        lis.append(rowlist)
    return lis
# -----------------------------------------------------------------------------------------
# function that makes a logical OR List operation
# input: list 1, list 2
# output: list of True/False status
def logic_or_list(lis1,lis2):
    lis = []
    for i in range(0,len(lis1)):
        rowlist = []
        for j in range(0,len(lis1[i])):
            if lis1[i][j] == False and lis2[i][j] == False:
                rowlist.append(False)
            else:
                rowlist.append(True)
        lis.append(rowlist)
    return lis
# -----------------------------------------------------------------------------------------
# function that makes a logical Not List operation
# input: list
# output: list of True/False status
def logic_not(lis1):
    lis = []
    for i in range(0,len(lis1)):
        rowlist = []
        for j in range(0,len(lis1[i])):
            rowlist.append(not lis1[i][j])
        lis.append(rowlist)
    return lis
# -----------------------------------------------------------------------------------------
# function that makes a apply mask on image
# input: skin_image, mask
# output: lmasked_skin
def mask_image(img, mask, value):
    for i in range(0,len(mask)):
        for j in range(0,len(mask[i])):
            if mask[i][j]:
                img[i][j] = value
    return img



