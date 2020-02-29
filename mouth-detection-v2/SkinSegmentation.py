import cv2
import numpy as np
import skimage.io as io

#----------------------------------------------------------------------------
# function used to exract skin from image
# input: image
# output: skin mask
def segmentSkin(img):
    B=img[:,:,0]
    G=img[:,:,1]
    R=img[:,:,2]

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]

    Y=0.299*R+0.587*G+0.114*B
    Cb=(B-Y)*0.564+128
    Cr=(R-Y)*0.713+128

    skin_one = np.zeros((len(img),len(img[0])))
    skin_one_h = np.logical_and(h>=0.0,h<=50.0)
    skin_one_s = np.logical_and(s>=0.23,s<=0.68)
    skin_one_rgb =  np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(R>95,G>40),B>20),R>G),R>B),np.absolute(np.array(R)-np.array(G)) > 15)
    skin_one = np.logical_and(np.logical_and(skin_one_h,skin_one_rgb),skin_one_s)

    skin_two = np.zeros((len(img),len(img[0])))
    skin_two_rgb = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(R>95,G>40),B>20),R>G),R>B),np.absolute(np.array(R)-np.array(G))>15)
    skin_two_YCbCr = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(Cr>135,Cb>85),Y>80),Cr<=((1.5862*Cb)+20)),Cr>=((0.3448*Cb)+76.2069)),
                                    Cr>=((-4.5652*Cb)+234.5652)),Cr<=((-1.15*Cb)+301.75)),Cr<=((-2.2857*Cb)+432.85))
    skin_two = np.logical_and(skin_two_YCbCr,skin_two_rgb)
    skin = np.logical_or(skin_one,skin_two)
    skin_image = np.copy(img)
    skin_image[skin] = 255
    skin_image[np.logical_not(skin)] = 0
    holes_filled_skin_image = fill_holes(skin_image)
    holes_filled_skin_image = holes_filled_skin_image.astype(np.uint8)
    holes_filled_skin_image = cv2.cvtColor(holes_filled_skin_image,cv2.COLOR_RGB2GRAY)
    cv2.imshow('skin_image', holes_filled_skin_image)
    return holes_filled_skin_image
#----------------------------------------------------------------------------
# function used to support segmentSkin function by filling holes in the skin mask from image
# input: skin mask
# output: skin mask
def fill_holes(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    res = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    return res
#----------------------------------------------------------------------------
# function used to exract skin from image
# input: image, skin mask
# output: skin only image
def extractSkin(img, skinMask):
    BLUR = 21
    img2 = np.copy(img)
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0,0,0)
    edges = cv2.Canny(skinMask, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    contour_info = []
    contours,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cntf = []
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    cv2.drawContours(img, contours, -1, 255, 3)
    cv2.imshow('contours', img) 
    cv2.waitKey(0)
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    #--------------------------
    max_contour = contour_info[0]

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    mask_stack = np.dstack([mask]*3)

    mask_stack  = mask_stack.astype('float32') / 255.0          
    img = img.astype('float32') / 255.0            

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) 
    masked = (masked * 255).astype('uint8')              
    return masked
