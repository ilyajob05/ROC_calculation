# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class FaceROCHelper:
    def __init__(self, idList, featureList, compareFunc=None):
        # initilisation
        assert(idList.shape[0] == featureList.shape[0])

        # ROC curve
        # index  0 - TPR, 1 - FPR, 2 - threshold
        self.curveROC = np.empty([0,3])

        self.idList = idList
        self.featureList = featureList
        if compareFunc is not None:
            self.compareFunc = compareFunc
        else:
            self.compareFunc = self.compareLen

        # fill id
        self.idNum = np.ndarray(self.idList.shape[0], np.uint)
        for i, inStr in enumerate(self.idList):
            self.idNum[i] = int(inStr[-3:])

        ###### allocate memory ######
        # matrix coeffs
        self.coeffMatrix = np.ndarray([self.featureList.shape[0], self.featureList.shape[0]], np.float32)
        # actual matrix covariation
        self.coeffMatrixReal = np.ndarray([self.featureList.shape[0], self.featureList.shape[0]], np.bool)
        # calculate matrix covariation
        self.coeffMatrixCompute = np.ndarray([self.featureList.shape[0], self.featureList.shape[0]], np.bool)
        # data buffer float
        self.buffSumm = np.ndarray([self.featureList.shape[0], 160], np.float16)
        # data buffer bool
        self.buffMask = np.ndarray([self.featureList.shape[0], 160], np.bool)

        # fill coeffs index
        np.equal.outer(self.idNum, self.idNum, out=self.coeffMatrixReal)
        self.P = (np.sum(self.coeffMatrixReal) - self.featureList.shape[0]) / 2.0 # correcting - remove comparisons of the same vectors and repeats
        self.N = (((self.featureList.shape[0]**2) - self.featureList.shape[0]) / 2.0) - self.P

        # calculation coeffs index
        # todo: remove repeat compare
        for i in range(self.featureList.shape[0]):
            self.compareFunc(self.featureList[i], self.featureList, self.coeffMatrix[i])

    # function for compare data from vectors
    def compareLen(self, feature, featureList, outArray):
        np.multiply(featureList, feature, out=self.buffSumm)
        self.buffSumm.sum(axis=1, out=outArray)

    # function to get TPR и FPR
    def getTPRFRP(self, threshold):
        # actual threshold
        np.greater(self.coeffMatrix, threshold, out=self.coeffMatrixCompute)

        # all positive var
        ALLPR = (np.sum(self.coeffMatrixCompute) - self.featureList.shape[0]) / 2.0  # correcting - remove comparisons of the same vectors and repeats

        # TP
        np.bitwise_and(self.coeffMatrixCompute, self.coeffMatrixReal, out=self.coeffMatrixCompute)
        TPSumm = (np.sum(self.coeffMatrixCompute) - self.featureList.shape[0]) / 2.0 # correcting - remove comparisons of the same vectors and repeats
        TPR = TPSumm / self.P

        # FP
        FPSumm = ALLPR - TPSumm
        FPR = FPSumm / self.N

        return [TPR, FPR]

    # get ROC curve
    # index  0 - TPR,    1 - FPR,    2 - threshold
    # todo: apply range of values
    def getCurve(self, step):
        curveROC = []
        subStep = int(1.0 / step)
        for i in range(subStep):
            currentThreshold = step * i
            curveROC.append(faceROC.getTPRFRP(currentThreshold) + [currentThreshold])
        self.curveROC = np.array(curveROC)
        return self.curveROC

    # find values for nearest value FPR
    def find2FPR(self, fpr):
        assert(self.curveROC.shape[0] > 0)
        idx = (np.abs(self.curveROC[:,1] - fpr)).argmin()
        return self.curveROC[idx]

    # show graph ROC
    def drawROC(self):
        assert (self.curveROC.shape[0] > 0)
        x = self.curveROC[:, 1]  # FPR
        y = self.curveROC[:, 0]  # TPR
        plt.plot(x, y)
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.title('ROC curve')
        plt.show()


if __name__ =='__main__':

    # load data
    f = np.load('/home/ilya/projects/testPyCharm/features.npy')
    id = np.load('/home/ilya/projects/testPyCharm/person_id.npy')

    print('data load...')
    faceROC = FaceROCHelper(id, f)
    print('complete')

    # calculation ROC to step 1e-1
    curveROC = faceROC.getCurve(1e-1)
    # show value
    print(curveROC)
    # draw graph
    faceROC.drawROC()


    # calculation ROC in step 1e-3 
    curveROC = faceROC.getCurve(1e-3)

    # return TPR and threshold for given FPR
    TPR_FPR_TH = faceROC.find2FPR(0.01)
    print('TPR: {}  FPR: {}  THRESHOLD: {}'.format(TPR_FPR_TH[0], TPR_FPR_TH[1], TPR_FPR_TH[2]))

    # show graph
    faceROC.drawROC()





