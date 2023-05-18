import numpy as np


class Grid:
    def __init__(self, m, block=True):
        if block:
            mb = m
            self.mb = mb
            self.m = 3 * mb + 1
        else:
            self.m = m
            if m % 3 == 1:
                self.mb = (m - 1) // 3
        self.N = self.m ** 2
        self.xvec, self.h = np.linspace(0, 1, self.m, retstep=True)
        self.yvec = np.linspace(0, 1, self.m)
        self.X, self.Y = np.meshgrid(self.xvec, self.yvec, indexing='ij')
        self.x = self.X.reshape((self.N, 1))
        self.y = self.Y.reshape((self.N, 1))
        self.xy = np.hstack((self.x, self.y))
        self.shape = (self.m, self.m)

    def mnh(self):
        """Returns m, N, h"""
        return self.m, self.N, self.h

    def params(self):
        """Returns m, N, h, X, Y, x, y"""
        return self.m, self.N, self.h, self.X, self.Y, self.x, self.y

    def get_hor_line(self, v, y):
        i = np.abs(self.yvec - y).argmin()
        return v.reshape(self.shape)[:, i]
