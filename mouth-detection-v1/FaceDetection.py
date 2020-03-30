import cv2
import numpy as np
import skimage.io as io

#----------------------------------------------------------------------------
# function used to get box of the face from image
# input: image
# output: image
def get_box(img,RGB_img):
    bounding_boxes = []
    masks = np.zeros(img.shape).astype("uint8")
    output = cv2.connectedComponentsWithStats(img, 8 , cv2.CV_32S)
    index = 0
    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    if num_labels>0:
        box_ratio_upper_bound = 1.1
        box_ratio_lower_bound = 0.4
        area=0
        index=0
        for i in range(0, labels.max()+1):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area=stats[i,cv2.CC_STAT_AREA]
            if x+w < y+h:
                minor_axis = x+w
                major_axis = y+h
            else:
                minor_axis = y+h
                major_axis = x+w
            if((w/h < box_ratio_upper_bound)and (w/h > box_ratio_lower_bound) and (minor_axis/major_axis > 0.25) and (minor_axis/major_axis<0.97) and area > 1500):
                if w/h <= 0.7:
                    h = int(h- 0.3*h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                mask = np.zeros(img.shape).astype("uint8")
                mask[y:y+h,x:x+w] = 255
                bounding_boxes.append([y,y+h,x,x+w])
                if  not((mask == 255).all()):
                    masks+= mask
                index+=1
        masks//=255
        masks[masks==1] = 255
        masks[masks>0] = 255
        img[masks==0] = 0
        cv2.imshow('skin_img', img) 
        cv2.waitKey(0)
        return bounding_boxes, img

def draw_RGB_with_Rect(RGB_image,Boundary_boxes, cp):
    ret, thresh = cv2.threshold(cp, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, width, height = cv2.boundingRect(contours[0])
    y0=y
    y1=y+height
    x0 = x
    x1= x+width
    cv2.rectangle(RGB_image,(x0,y0),(x1,y1),(255,0,0),2)
    # roi = RGB_image[y:y+height, x:x+width]
    return RGB_image,y0,y1,x0,x1