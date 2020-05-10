import numpy as np


def extractBGRChannels(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    return R, G, B


def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return h, s, v


def getHSV(img):
    R, G, B = extractBGRChannels(img)
    H = np.zeros((R.shape[0], R.shape[1]))
    S = np.zeros((R.shape[0], R.shape[1]))
    V = np.zeros((R.shape[0], R.shape[1]))
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            H[i][j], S[i][j], V[i][j] = rgb_to_hsv(R[i][j], G[i][j], B[i][j])

    HSV = np.zeros((R.shape[0], R.shape[1], 3), 'uint8')
    HSV[..., 0] = H
    HSV[..., 1] = S
    HSV[..., 2] = V
    return H, S, V
