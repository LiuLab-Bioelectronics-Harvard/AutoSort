import math
import scipy
import numpy as np


def offline_smoother(yIn, kernSD, stepSize):
    causal = False

    if (kernSD == 0) or (yIn.shape[1] == 1):
        yOut = yIn
        return yOut

    # Filter half length
    # Go 3 standard deviations out
    fltHL = math.ceil(3 * kernSD / stepSize)

    # Length of flt is 2*fltHL + 1
    flt = scipy.stats.norm.pdf(
        np.arange(-fltHL * stepSize, fltHL * stepSize, stepSize), loc=0, scale=kernSD
    )

    [yDim, T] = shape(yIn)
    yOut = np.zeros((yDim, T))

    nm = scipy.signal.convolve(
        flt.reshape(1, -1), np.ones((1, T)), mode="full", method="auto"
    )

    for i in range(yDim):

        ys = np.divide(
            scipy.signal.convolve(flt.reshape(1, -1), yIn[i, :].reshape(1, -1)), nm
        )
        #     # Cut off edges so that result of convolution is same length
        #     # as original data
        yOut[i, :] = ys[0, fltHL : ys.shape[1] - fltHL + 1]
    return yOut


class FSM2:
    def __init__(self, k, d, tau0=None, Minv0=None, W0=None, learning_rate=None):
        self.k = k
        self.d = d
        if W0 is None:
            self.W = np.eye(k, d) / d
        else:
            self.W = W0
        self.M = np.eye(k)
        if Minv0 is None:
            self.Minv = np.eye(k)
        else:
            self.Minv = Minv0
        if tau0 is None:
            self.tau = 0.5
        else:
            self.tau = tau0
        if learning_rate is None:
            self.lr = lambda x: 1.0 / (2 * (x + 100) + 5)
        else:
            self.lr = lambda x: learning_rate

        self.t = 0
        self.outer_W = np.zeros(shape(self.W))
        self.outer_Minv = np.zeros(shape(self.Minv))
        self.y = []

    def fit_next(self, x):
        step = self.lr(self.t)
        #         print('step',step)

        z = np.dot(self.W, x.T)
        y = np.zeros(shape(z))
        #         print('z',z)
        #         print('y',y)

        # The loop below solves the linear system My = Wx
        y[0] = z[0] / self.M[0, 0]
        #         print('y[0]',y[0])
        for i in np.arange(1, self.k):
            y[i] = z[i]
            #             print('i',i,'y[i]',y[i])
            for j in np.arange(0, i):
                y[i] = y[i] - y[j] * self.M[i, j]
            #                 print('i',i,'j',j,'y[i]',y[i])

            y[i] = y[i] / self.M[i, i]
        #             print('i',i,'y[i]',y[i])

        y_up = np.tril(
            np.dot(y.reshape(y.shape[0], 1), y.reshape(y.shape[0], 1).T)
        )  # extract lower triangular component of y
        #         print('y_up',y_up)
        self.W = self.W + step * (
            y.reshape(y.shape[0], 1) * x.reshape(x.shape[0], 1).T - self.W
        )  # update W matrix
        #         print('W',self.W)
        self.M = self.M + step * (y_up - self.M)  # update M matrix
        #         print('M',self.M)
        self.t = self.t + 1
        #         print('t',self.t)
        return y
