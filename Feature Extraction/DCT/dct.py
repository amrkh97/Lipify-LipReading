import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Constructing DCT using Quantizatoin matrix
############# Failure attempt !! ################

quantization_matrix = [
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
]

cosine_table = [
  [math.cos((2*i+1)*j * math.pi/16) for j in range(8)] for i in range(8)
]

partition_list = [(i,j) for i in range(8) for j in range(8)]


def DCT (a,u,v):
      value = 0
      for i,j in partition_list:
            value += a[i][j] * cosine_table[i][u] * cosine_table[j][v]
      if u == 0: value *= (1/math.sqrt(2))
      if v == 0: value *= (1/math.sqrt(2))
      value *= 0.25
      return value




def reduce_128 (pixel):
  pixel -= 128
  return pixel


def Encode (img):
  img = [[reduce_128(pixel) for pixel in row] for row in img]
  img = [[DCT(img,u,v) for v in range(8)] for u in range(8) ]
  img = [[round(a/q) for a,q in zip(a,q)] for a,q in zip (img,quantization_matrix)]
  return img

############################################################################################################
############################################################################################################
############################################################################################################
############################### Constructing DCT using O(n^4) loops ########################################
############################################################################################################
############################################################################################################
############################################################################################################

############# Second Failure attempt !! ################

def DCT_cosine (img_index, dct_index, img_row_or_col_size):
  return np.cos(math.radians((2*img_index+1)*dct_index*np.pi/2*img_row_or_col_size))


def constant_value (dct_index, img_row_or_col_size):
  if dct_index == 0:
    return np.sqrt(1/img_row_or_col_size)
  else:
    return np.sqrt(2/img_row_or_col_size)

def DCT_for_one_pixel (img, dct_row_index, dct_col_index, img_row_size, img_col_size):
      value = 0;
      for img_row_index in range(img_row_size):
            for img_col_index in range(img_col_size):
                  value += img[img_row_index,img_col_index] * DCT_cosine(img_row_index,dct_row_index,img_row_size)* DCT_cosine(img_col_index,dct_col_index,img_col_size)
      return value


def DCT_Matrix (D, img):
      for u in range(D.shape[0]):
          for v in range(D.shape[1]):
            D[u,v] = constant_value(u, D.shape[0]) * constant_value(v, D.shape[1]) * DCT_for_one_pixel(img,u,v,img.shape[0],img.shape[1])
            D[u,v] = DCT_for_one_pixel(img,u,v,img.shape[0],img.shape[1])

      return D


############################################################################################################
############################################################################################################
############################################################################################################
############################### Constructing DCT using basic matrix ########################################
############################################################################################################
############################################################################################################
############################################################################################################


def basic_matrix(N):
      C = np.zeros((N,N))
      C[0,] = np.sqrt(1/N)

      for u in range(1,N):
            for v in range(N):
                  C[u,v] = np.sqrt(2/N) *math.cos(((2*v+1)*np.pi*u)/(2*N))
      return C

image = cv2.imread('/home/abdelgawad/Folders/Graduation-Project/Feature Extraction/DCT/panda.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image =  np.float32(image)/255.0


C = basic_matrix(image.shape[0])
temp = C.dot(image)
D = temp.dot(np.transpose(C))
print(D)

print("-------------------------------------------------------")
#test against openCv DCT function
dct = cv2.dct(image)
print(dct)


print(np.allclose(D,dct,atol=1e-05)) # hand made DCT gets the same output as the ready made one with tolreance of 10^-5

