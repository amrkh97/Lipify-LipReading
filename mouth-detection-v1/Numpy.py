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
# function that makes a logical AND operation
# input: matrix 1, matrix 2
# output: matrix of True/False status
def logic_and(mat1, mat2):
    mat = []
    for i in range(0, mat1.shape[0]):
        rowlist = []
        for j in range(0, mat1.shape[1]):
            if mat1[i][j] > 0 and  mat2[i][j] > 0:
                rowlist.append(True)
            else:
                rowlist.append(False)
        mat.append(rowlist)
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
# function that makes a logical OR operation
# input: matrix 1, matrix 2
# output: matrix of True/False status
def logic_or(mat1, mat2):
    mat = []
    for i in range(0, mat1.shape[0]):
        rowlist = []
        for j in range(0, mat1.shape[1]):
            if mat1[i][j] == 0 and  mat2[i][j] == 0:
                rowlist.append(False)
            else:
                rowlist.append(True)
        mat.append(rowlist)
    return mat
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

