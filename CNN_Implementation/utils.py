import gzip

import numpy as np
from CNN_Implementation import forward


def extract_data(filename, num_images, IMAGE_WIDTH):
    """
    Function to extract images by reading the file bytestream.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH * IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    """
    Function to extract label into vector of integer values.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def initializeFilter(size, scale=1.0):
    """
    Function to initialize the filter with random values.
    """
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initializeWeight(size):
    """
    Function to initialize weights of layers.
    """
    return np.random.standard_normal(size=size) * 0.01


def nanargmax(arr):
    """
    Function to return indices of max elements in the array.
    """
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs


def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s=1, pool_f=2, pool_s=2):
    """
    Function to make predictions with trained weights.
    """

    conv1 = forward.convolution(image, f1, b1, conv_s)
    conv1[conv1 <= 0] = 0  # ReLU activation

    conv2 = forward.convolution(conv1, f2, b2, conv_s)
    conv2[conv2 <= 0] = 0  # ReLU activation

    pooled = forward.maxpool(conv2, pool_f, pool_s)
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = w3.dot(fc) + b3  # Dense layer
    z[z <= 0] = 0  # ReLU activation

    out = w4.dot(z) + b4  # Dense layer
    probs = forward.softmax(out)  # softmax activation
    return np.argmax(probs), np.max(probs)
