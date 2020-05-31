import FisherFace
import feature
import test
faces, testLabel = FisherFace.read_faces('test')
z, Ye, Yt = feature.PCA(faces)

print('the accuracy of PCA:')
test.accur(z, Yt)


