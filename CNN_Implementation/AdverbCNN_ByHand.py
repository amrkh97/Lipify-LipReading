import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from CNN_Implementation import backward, changeImagesToMnistFormat, forward, utils

AdverbDict = {0: 'again', 1: 'now', 2: 'please', 3: 'soon'}


def conv(image, label, params, conv_s, pool_f, pool_s):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    conv1 = forward.convolution(image, f1, b1, conv_s)
    conv1[conv1 <= 0] = 0  # ReLU activation

    conv2 = forward.convolution(conv1, f2, b2, conv_s)
    conv2[conv2 <= 0] = 0  # ReLU activation

    pooled = forward.maxpool(conv2, pool_f, pool_s)

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))

    z = w3.dot(fc) + b3  # Dense layer
    z[z <= 0] = 0  # ReLU activation

    out = w4.dot(z) + b4  # Dense layer

    probs = forward.softmax(out)  # Softmax activation
    # Loss Calculation:
    loss = forward.categoricalCrossEntropy(probs, label)
    # BackPropagation:
    dout = probs - label
    dw4 = dout.dot(z.T)
    db4 = np.sum(dout, axis=1).reshape(b4.shape)

    dz = w4.T.dot(dout)
    dz[z <= 0] = 0  # ReLU activation
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis=1).reshape(b3.shape)

    dfc = w3.T.dot(dz)
    dpool = dfc.reshape(pooled.shape)

    dconv2 = backward.maxpoolBackward(dpool, conv2, pool_f, pool_s)
    dconv2[conv2 <= 0] = 0  # ReLU Activation

    dconv1, df2, db2 = backward.convolutionBackward(dconv2, conv1, f2, conv_s)

    dconv1[conv1 <= 0] = 0  # ReLU Activation

    dimage, df1, db1 = backward.convolutionBackward(dconv1, image, f1, conv_s)

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss


def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    """
    update the parameters through Adam gradient descent.
    """
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    X = batch[:, 0:-1]  # Inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:, -1]  # Labels

    cost_ = 0
    batch_size = len(batch)

    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)

    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)

    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)

    for i in range(batch_size):
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # One-Hot Encoding

        grads, loss = conv(x, y, params, 1, 2, 2)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

        df1 += df1_
        db1 += db1_
        df2 += df2_
        db2 += db2_
        dw3 += dw3_
        db3 += db3_
        dw4 += dw4_
        db4 += db4_

        cost_ += loss

    # Parameter Update:
    v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size  # momentum update
    s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2  # RMSProp update
    f1 -= lr * v1 / np.sqrt(s1 + 1e-7)  # Adam update: momentum & RMSProp

    bv1 = beta1 * bv1 + (1 - beta1) * db1 / batch_size
    bs1 = beta2 * bs1 + (1 - beta2) * (db1 / batch_size) ** 2
    b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

    v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
    s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
    f2 -= lr * v2 / np.sqrt(s2 + 1e-7)

    bv2 = beta1 * bv2 + (1 - beta1) * db2 / batch_size
    bs2 = beta2 * bs2 + (1 - beta2) * (db2 / batch_size) ** 2
    b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

    v3 = beta1 * v3 + (1 - beta1) * dw3 / batch_size
    s3 = beta2 * s3 + (1 - beta2) * (dw3 / batch_size) ** 2
    w3 -= lr * v3 / np.sqrt(s3 + 1e-7)

    bv3 = beta1 * bv3 + (1 - beta1) * db3 / batch_size
    bs3 = beta2 * bs3 + (1 - beta2) * (db3 / batch_size) ** 2
    b3 -= lr * bv3 / np.sqrt(bs3 + 1e-7)

    v4 = beta1 * v4 + (1 - beta1) * dw4 / batch_size
    s4 = beta2 * s4 + (1 - beta2) * (dw4 / batch_size) ** 2
    w4 -= lr * v4 / np.sqrt(s4 + 1e-7)

    bv4 = beta1 * bv4 + (1 - beta1) * db4 / batch_size
    bs4 = beta2 * bs4 + (1 - beta2) * (db4 / batch_size) ** 2
    b4 -= lr * bv4 / np.sqrt(bs4 + 1e-7)

    cost_ = cost_ / batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    return params, cost


def checkPreTrained(file_path):
    if os.path.exists(file_path):
        return True
    return False


def train(num_classes=4, lr=0.001, beta1=0.95, beta2=0.99, img_dim=224, img_depth=1, f=5, num_filt1=8, num_filt2=8,
          batch_size=1, num_epochs=10, save_path='params.pkl'):
    m = 250  # Train on 250 examples only.
    X = utils.extract_data('../Compressed-Dataset/adverb-images-idx3-ubyte.gz', m, img_dim)
    y_dash = utils.extract_labels('../Compressed-Dataset/adverb-labels-idx1-ubyte.gz', m).reshape(m, 1)
    X -= int(np.mean(X))
    X /= int(np.std(X))
    train_data = np.hstack((X, y_dash))

    np.random.shuffle(train_data)

    # Initialize parameters:
    # 93312: 8 * 108 * 108 (After MaxPool Layer and Flatten)
    f1, f2, w3, w4 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (128, 93312), (num_classes, 128)
    b1, b2, b3, b4 = 0, 0, 0, 0
    params = []
    cost = []
    if not checkPreTrained(save_path):
        f1 = utils.initializeFilter(f1)
        f2 = utils.initializeFilter(f2)
        w3 = utils.initializeWeight(w3)
        w4 = utils.initializeWeight(w4)

        b1 = np.zeros((f1.shape[0], 1))
        b2 = np.zeros((f2.shape[0], 1))
        b3 = np.zeros((w3.shape[0], 1))
        b4 = np.zeros((w4.shape[0], 1))

        params = [f1, f2, w3, w4, b1, b2, b3, b4]
    else:
        params, cost = pickle.load(open(save_path, 'rb'))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))

    to_save = [params, cost]

    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)

    return cost


def trainAdverbByHandCNN(parametersSavePath, num_epochs=10):
    cost = train(num_epochs=num_epochs, save_path=parametersSavePath)

    params, cost = pickle.load(open(parametersSavePath, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # Plot cost
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()
    print(params)


def predictAdverb(image, save_path):
    changeImagesToMnistFormat.compressOneImage(image)

    X = utils.extract_data('Test-image-idx3-ubyte.gz', 1, 224)
    X -= int(np.mean(X))
    X /= int(np.std(X))
    X = X.reshape(1, 224, 224)
    os.remove('Test-image-idx3-ubyte.gz')

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    pred, prob = utils.predict(X, f1, f2, w3, w4, b1, b2, b3, b4)
    return AdverbDict[pred], prob


if __name__ == "__main__":
    trainAdverbByHandCNN('params.pkl', num_epochs=10)

    image_path = 'C:/Users/amrkh/Desktop/CNN-Test-Images/Adverb/please/0.jpg'
    image_path = Image.open(image_path)

    print(predictAdverb(image_path, 'params.pkl'))
