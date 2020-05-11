import cv2
import numpy as np

def getSkin(img):
    img_corrected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    R = img_corrected[:, :, 0]
    G = img_corrected[:, :, 1]
    B = img_corrected[:, :, 2]

    skin = np.zeros((len(img), len(img[0])))

    for i in range(img_corrected.shape[0]):
        for j in range(img_corrected.shape[1]):
            if ((R[i][j] > 95) and (G[i][j] > 40) and (B[i][j] > 20) and ((max([R[i][j], G[i][j], B[i][j]]) - min([R[i][j], G[i][j], B[i][j]])) > 15) and 
            (abs((int(R[i][j]) - int(G[i][j]))) > 15) and (R[i][j] > G[i][j]) and (R[i][j] > B[i][j])):
                cond1 = (R[i][j]/(int(R[i][j]) + int(G[i][j])+ int(B[i][j])))
                cond2 = (G[i][j]/(int(R[i][j]) + int(G[i][j])+ int(B[i][j])))
                if (0.36 <= cond1 and cond1 <= 0.465 and 0.28 <= cond2 and cond2 <= 0.363):
                    skin[i][j] = 1
                else:
                    skin[i][j] = 0
            else:
                skin[i][j] = 0
    return skin
