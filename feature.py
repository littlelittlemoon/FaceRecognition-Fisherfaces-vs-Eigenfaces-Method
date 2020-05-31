import FisherFace
import numpy as np

faceMatrix, label = FisherFace.read_faces('train')

def compute(K, faces):
    [r, c] = faceMatrix.shape
    W, LL, m = FisherFace.myPCA(faceMatrix)
    W1 = W[:,:K]
    facesNew = faces - np.transpose(np.tile(m, (c, 1)))
    X = np.transpose(np.dot(np.transpose(facesNew), W1))
    return X

#PCA
def PCA(faces):
    K = 30
    Ye = compute(K, faceMatrix)
    [r, c] = Ye.shape
    x = []
    z = np.ones((10,K))
    for i in range(c):
        x.append(Ye[:, i])
        if (i + 1) % 12 == 0:
            temp = np.array(x).transpose()
            j = i // 12
            z[j] = np.mean(temp, 1)
            x = []
    z = z.transpose()
    Yt = compute(K, faces)
    return z, Ye, Yt

#LDA
def LDA(faces):
    K1 = 90
    X = compute(K1, faceMatrix)
    Wf, C, classLabels = FisherFace.myLDA(X, label)
    Y = compute(K1, faces)
    Y = np.dot(Wf.transpose(), Y)
    Yf = np.dot(Wf.transpose(), X)
    return C, Yf, Y

