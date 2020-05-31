import FisherFace
import feature
from matplotlib import pyplot
from pylab import *
import test
import numpy as np

test_faces, testLabel = FisherFace.read_faces('test')
train_faces, label = FisherFace.read_faces('train')

def fused(faces, alpha):
    z, Ye, Yt = feature.PCA(faces)
    C, Yf, Y  = feature.LDA(faces)

    a = np.dot(alpha, Ye)
    b = np.dot((1 - alpha), Yf)
    y = np.concatenate((a, b))
    return y

def compute(y):
    x = []
    l = y.shape
    z = np.ones((10,39))
    for i in range(l[1]):
        x.append(y[:,i])
        if (i+1) % 12 == 0:
            temp = np.array(x).transpose()
            j = i // 12
            z[j] = np.mean(temp, 1)
            x = []
    z = z.transpose()
    return z

for i in range(1,10):
    alpha = i / 10
    print('alpha = ', alpha)

    y = fused(train_faces, alpha)
    z = compute(y)
    Yt = fused(test_faces, alpha)

    plot = test.accur(z, Yt)
    pyplot.plot(alpha, plot, 'o')

pyplot.savefig('result.jpg')
show()
