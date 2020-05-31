import FisherFace
import feature
import test

faces, testLabel = FisherFace.read_faces('test')
C, Yf, Y  = feature.LDA(faces)
print('the accuracy of LDA:')
test.accur(C, Y)
