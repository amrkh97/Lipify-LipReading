import numpy as np


def convolution(image, filt, bias, s=1):
    """
    Convolves `filt` over `image` using stride `s`
    """
    (n_f, n_c_f, f, _) = filt.shape
    n_c, in_dim, _ = image.shape

    out_dim = int((in_dim - f) / s) + 1

    out = np.zeros((n_f, out_dim, out_dim))

    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                            bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out


def maxpool(image, f=2, s=2):
    """
    Apply max pool filter on image
    """
    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled


def softmax(X):
    """
    Function to apply Softmax activation
    """
    out = np.exp(X)
    return out / np.sum(out)


def categoricalCrossEntropy(probs, label):
    """
    Function to calculate catgorical cross entropy loss
    """
    return -np.sum(label * np.log(probs))
