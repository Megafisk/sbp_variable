# Demonstrates how to construct 2D variable coefficient second-derivative SBP operators, and verifies SBP properties.
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplg
from D2_Variable import *
# import orig.test_orig as ot

# Number of grid points
m = 31
N = m * m

# Space discretization
xvec, h = np.linspace(0, 1, m, retstep=True)
yvec, h = np.linspace(0, 1, m, retstep=True)
[X, Y] = np.meshgrid(xvec, yvec, indexing='ij')
# x = np.reshape(X.T, [N])
# y = np.reshape(Y.T, [N])

# Variable coefficient, some positive function of x and y
b = np.sqrt(X * Y + 1) * np.exp(np.cos(X) ** np.sin(Y))
bb = b.reshape((1, N))[0]

ops_1d = D2_Variable_4(m, h)
H, HI, D1, D2_fun, e_l, e_r, d1_l, d1_r = ops_1d
HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = ops_2d(m, h, bb, ops_1d)

# Check SBP property. Eigenvalues should be real and non-positive.
Mx = -HH @ D2x - spsp.diags(bb) @ eW @ H @ d1_W.T + spsp.diags(bb) @ eE @ H @ d1_E.T
My = -HH @ D2y - spsp.diags(bb) @ eS @ H @ d1_S.T + spsp.diags(bb) @ eN @ H @ d1_N.T

[eigMx, _] = spsplg.eigs(Mx)
print(np.max(np.max(np.abs(Mx - Mx.T))))
print(np.max(np.abs(np.imag(eigMx))))
print(np.min(np.real(eigMx)))

[eigMy, _] = spsplg.eigs(My)
print(np.max(np.max(np.abs(My - My.T))))
print(np.max(np.abs(np.imag(eigMy))))
print(np.min(np.real(eigMy)))
