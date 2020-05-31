import FisherFace
import feature
import numpy as np

faces, test_label = FisherFace.read_faces('test')

def accur(y1, y2):
    num = y1.shape
    num_t = y2.shape
    confusion = np.zeros((10, 10))
    dis = []
    accuracy = 0
    for i in range(0, num_t[1]):
        for j in range(0, num[1]):
            dis.append(np.linalg.norm(y2[:,i] - y1[:,j]))
        minDis = min(dis)
        index = dis.index(minDis)
        dis = []
        label = test_label[i]
        if (label == index):
            confusion[label][label] += 1
            accuracy += 1
        else:
        	confusion[label][index] += 1
    accuracy = accuracy / num_t[1] * 100
    
    print('%.2f %%' %accuracy)
    print('confusion matrix:\n', confusion)
    return accuracy

# fusion feature
def fused(faces, Alpha):
    z, Ye, Yt = feature.PCA(faces)
    C, Yf, Y  = feature.LDA(faces)

    a = np.dot(Alpha, Ye)
    b = np.dot((1 - Alpha), Yf)
    y = np.concatenate((a, b))
