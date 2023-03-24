import numpy as np
import scipy.linalg as splg
import scipy.sparse as spsp


######################################################################################
##                                                                                  ##
##  SBP operators for course "Scientific computing for PDEs" at Uppsala University. ##
##                                                                                  ##
##  Author: Gustav Eriksson                                                         ##
##  Date:   2022-08-31                                                              ##
##                                                                                  ##
##  Based on Matlab code written by Ken Mattsson.                                   ##
##                                                                                  ##
##  Central operators of orders 2, 4, and 6.                                        ##
##  Upwind operators of order 3, 5, and 7.                                          ##
##  Periodic explicit operators of order 2, 4, 6, 8, 10, and 12.                    ##
##  Periodic implicit operators.                                                    ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##  - Scipy      1.7.0                                                              ##
##  - Matplotlib 3.3.4                                                              ##
##                                                                                  ##
######################################################################################

def convert_1d_2d(ops, m):
    """owo

    @param ops: (H, HI, D1, D2, e_l, e_r, d1_l, d1_r) from the functions in this class
    @param m: number of grid points
    @return: HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N)
    """
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops
    Im = spsp.eye(m)
    HH = spsp.kron(H, H, 'dia')
    HHI = spsp.kron(HI, HI, 'dia')
    eW = spsp.kron(Im, e_l, 'csr')
    eE = spsp.kron(Im, e_r, 'csr')
    eS = spsp.kron(e_l, Im, 'csr')
    eN = spsp.kron(e_r, Im, 'csr')
    d1_W = spsp.kron(Im, d1_l, 'csr')
    d1_E = spsp.kron(Im, d1_r, 'csr')
    d1_S = spsp.kron(d1_l, Im, 'csr')
    d1_N = spsp.kron(d1_r, Im, 'csr')
    D2x = spsp.kron(D2, Im, 'csr')
    D2y = spsp.kron(Im, D2, 'csr')
    return HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N)

# Central 1D second order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   D1 - first derivative SBP operator
#   D2 - second derivative SBP operator
#   e_l,e_r - vectors to extract the boundary grid points
#   d1_l,d1_r - vectors to extract the first derivatives at the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(m,h,order)
def sbp_cent_2nd(m, h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.eye(m)
    H[0, 0] = 0.5
    H[-1, -1] = 0.5
    H = h * H

    HI = np.linalg.inv(H)

    D1 = 0.5 * np.diag(np.ones(m - 1), 1) - 0.5 * np.diag(np.ones(m - 1), -1)
    D1[0, 0] = -1
    D1[0, 1] = 1
    D1[-1, -2] = -1
    D1[-1, -1] = 1
    D1 = D1 / h

    Q = np.matmul(H, D1) + 0.5 * np.tensordot(e_l, e_l, axes=0) - 0.5 * np.tensordot(e_r, e_r, axes=0)

    D2 = np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1) - 2 * np.diag(np.ones(m), 0)
    D2[0, 0] = 1
    D2[0, 1] = -2
    D2[0, 2] = 1
    D2[-1, -3] = 1
    D2[-1, -2] = -2
    D2[-1, -1] = 1
    D2 = D2 / (h * h)

    d_stenc = np.array([-3. / 2, 2, -1. / 2]) / h
    d1_l = np.zeros(m)
    d1_l[:3] = d_stenc
    d1_r = np.zeros(m)
    d1_r[-3:] = -np.flip(d_stenc)

    M = -np.matmul(H, D2) - np.tensordot(e_l, d1_l, axes=0) + np.tensordot(e_r, d1_r, axes=0)

    H = spsp.csc_matrix(H)
    HI = spsp.csc_matrix(HI)
    D1 = spsp.csc_matrix(D1)
    D2 = spsp.csc_matrix(D2)
    e_l = spsp.csc_matrix(e_l)
    e_r = spsp.csc_matrix(e_r)
    d1_l = spsp.csc_matrix(d1_l)
    d1_r = spsp.csc_matrix(d1_r)
    return H, HI, D1, D2, e_l, e_r, d1_l, d1_r


# Central 1D fourth order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   D1 - first derivative SBP operator
#   D2 - second derivative SBP operator
#   e_l,e_r - vectors to extract the boundary grid points
#   d1_l,d1_r - vectors to extract the first derivatives at the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(m,h,order)
def sbp_cent_4th(m, h):
    e_l = spsp.coo_array(([1], ([0], [0])), shape=(m, 1)).tocsr()
    e_r = spsp.coo_array(([1], ([m - 1], [0])), shape=(m, 1)).tocsr()

    H_diag = np.ones(m)
    H_diag[:4] = [17 / 48, 59 / 48, 43 / 48, 49 / 48]
    H_diag[-4:] = [49 / 48, 43 / 48, 59 / 48, 17 / 48]
    H_diag = H_diag * h
    H = spsp.diags(H_diag)
    HI = spsp.diags(1 / H_diag)

    Q = spsp.diags([-1 / 12, 8 / 12, -8 / 12, 1 / 12], [2, 1, -1, -2], (m, m)).tolil()
    Q_U = np.array([[0, 59 / 96, -1 / 12, -1 / 32], [-59 / 96, 0, 59 / 96, 0],
                    [1 / 12, -59 / 96, 0, 59 / 96], [1 / 32, 0, -59 / 96, 0]])
    Q[0:4, 0:4] = Q_U
    Q[-4:, -4:] = np.flipud(np.fliplr(-Q_U))
    Q = Q.tocsr()

    D1 = HI @ (Q - 1 / 2 * (e_l @ e_l.transpose()) + 1 / 2 * (e_r @ e_r.transpose()))

    M_U = np.array([[9 / 8, -59 / 48, 1 / 12, 1 / 48],
                    [-59 / 48, 59 / 24, -59 / 48, 0],
                    [1 / 12, -59 / 48, 55 / 24, -59 / 48],
                    [1 / 48, 0, -59 / 48, 59 / 24]])
    M = spsp.diags((1 / 12, -4 / 3, 5 / 2, -4/3, 1/12), (-2, -1, 0, 1, 2), (m, m)).tolil()
    M[0:4, 0:4] = M_U
    M[-4:, -4:] = np.flipud(np.fliplr(M_U))
    M = (M / h).tocsr()

    d_stenc = np.array([-11 / 6, 3, -3 / 2, 1 / 3]) / h

    d1_l = spsp.lil_array((m, 1))
    d1_l[0:4] = d_stenc
    d1_l = d1_l.tocsr()

    d1_r = spsp.lil_array((m, 1))
    d1_r[-4:] = np.flip(-d_stenc)
    d1_r = d1_r.tocsr()

    D2 = HI @ (-M - (e_l @ d1_l.T) + (e_r @ d1_r.T))

    return H, HI, D1, D2, e_l, e_r, d1_l, d1_r


# Central 1D sixth order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   D1 - first derivative SBP operator
#   D2 - second derivative SBP operator
#   e_l,e_r - vectors to extract the boundary grid points
#   d1_l,d1_r - vectors to extract the first derivatives at the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(m,h,order)
def sbp_cent_6th(m, h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m), 0);
    H[:6, :6] = np.diag(np.array([13649 / 43200, 12013 / 8640, 2711 / 4320, 5359 / 4320, 7877 / 8640, 43801 / 43200]))
    H[-6:, -6:] = np.fliplr(np.flipud(
        np.diag(np.array([13649 / 43200, 12013 / 8640, 2711 / 4320, 5359 / 4320, 7877 / 8640, 43801 / 43200]))));
    H = H * h;

    HI = np.linalg.inv(H)

    x1 = 0.70127127127127;

    D1 = 1 / 60 * np.diag(np.ones(m - 3), 3) - 9 / 60 * np.diag(np.ones(m - 2), 2) + 45 / 60 * np.diag(np.ones(m - 1),
                                                                                                       1) - 45 / 60 * np.diag(
        np.ones(m - 1), -1) + 9 / 60 * np.diag(np.ones(m - 2), -2) - 1 / 60 * np.diag(np.ones(m - 3), -3)

    D1_bound_stencil = np.array([[-21600 / 13649, 43200 / 13649 * x1 - 7624 / 40947,
                                  -172800 / 13649 * x1 + 715489 / 81894, 259200 / 13649 * x1 - 187917 / 13649,
                                  -172800 / 13649 * x1 + 735635 / 81894, 43200 / 13649 * x1 - 89387 / 40947, 0, 0, 0], \
                                 [-8640 / 12013 * x1 + 7624 / 180195, 0, 86400 / 12013 * x1 - 57139 / 12013,
                                  -172800 / 12013 * x1 + 745733 / 72078, 129600 / 12013 * x1 - 91715 / 12013,
                                  -34560 / 12013 * x1 + 240569 / 120130, 0, 0, 0], \
                                 [17280 / 2711 * x1 - 715489 / 162660, -43200 / 2711 * x1 + 57139 / 5422, 0,
                                  86400 / 2711 * x1 - 176839 / 8133, -86400 / 2711 * x1 + 242111 / 10844,
                                  25920 / 2711 * x1 - 182261 / 27110, 0, 0, 0], \
                                 [-25920 / 5359 * x1 + 187917 / 53590, 86400 / 5359 * x1 - 745733 / 64308,
                                  -86400 / 5359 * x1 + 176839 / 16077, 0, 43200 / 5359 * x1 - 165041 / 32154,
                                  -17280 / 5359 * x1 + 710473 / 321540, 72 / 5359, 0, 0], \
                                 [34560 / 7877 * x1 - 147127 / 47262, -129600 / 7877 * x1 + 91715 / 7877,
                                  172800 / 7877 * x1 - 242111 / 15754, -86400 / 7877 * x1 + 165041 / 23631, 0,
                                  8640 / 7877 * x1, -1296 / 7877, 144 / 7877, 0], \
                                 [-43200 / 43801 * x1 + 89387 / 131403, 172800 / 43801 * x1 - 240569 / 87602,
                                  -259200 / 43801 * x1 + 182261 / 43801, 172800 / 43801 * x1 - 710473 / 262806,
                                  -43200 / 43801 * x1, 0, 32400 / 43801, -6480 / 43801, 720 / 43801]])

    D1[:6, :9] = D1_bound_stencil
    D1[-6:, -9:] = np.flipud(np.fliplr(-D1_bound_stencil))
    D1 = D1 / h

    Q = np.matmul(H, D1) + 0.5 * np.tensordot(e_l, e_l, axes=0) - 0.5 * np.tensordot(e_r, e_r, axes=0)

    D2 = (2 * np.diag(np.ones(m - 3), 3) - 27 * np.diag(np.ones(m - 2), 2) + 270 * np.diag(np.ones(m - 1),
                                                                                           1) + 270 * np.diag(
        np.ones(m - 1), -1) - 27 * np.diag(np.ones(m - 2), -2) + 2 * np.diag(np.ones(m - 3), -3) - 490 * np.diag(
        np.ones(m), 0)) / 180;

    D2_bound_stencil = np.array(
        [[114170 / 40947, -438107 / 54596, 336409 / 40947, -276997 / 81894, 3747 / 13649, 21035 / 163788, 0, 0, 0], \
         [6173 / 5860, -2066 / 879, 3283 / 1758, -303 / 293, 2111 / 3516, -601 / 4395, 0, 0, 0], \
         [-52391 / 81330, 134603 / 32532, -21982 / 2711, 112915 / 16266, -46969 / 16266, 30409 / 54220, 0, 0, 0], \
         [68603 / 321540, -12423 / 10718, 112915 / 32154, -75934 / 16077, 53369 / 21436, -54899 / 160770, 48 / 5359, 0,
          0], \
         [-7053 / 39385, 86551 / 94524, -46969 / 23631, 53369 / 15754, -87904 / 23631, 820271 / 472620, -1296 / 7877,
          96 / 7877, 0], \
         [21035 / 525612, -24641 / 131403, 30409 / 87602, -54899 / 131403, 820271 / 525612, -117600 / 43801,
          64800 / 43801, -6480 / 43801, 480 / 43801]]);

    D2[:6, :9] = D2_bound_stencil
    D2[-6:, -9:] = np.flipud(np.fliplr(D2_bound_stencil))

    D2 = D2 / h ** 2

    d_stenc = np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]) / h
    d1_l = np.zeros(m)
    d1_l[0:5] = d_stenc
    d1_r = np.zeros(m)
    d1_r[-5:] = np.flip(-d_stenc)

    M = -np.matmul(H, D2) - np.tensordot(e_l, d1_l, axes=0) + np.tensordot(e_r, d1_r, axes=0)

    H = spsp.csc_matrix(H)
    HI = spsp.csc_matrix(HI)
    D1 = spsp.csc_matrix(D1)
    D2 = spsp.csc_matrix(D2)
    e_l = spsp.csc_matrix(e_l)
    e_r = spsp.csc_matrix(e_r)
    d1_l = spsp.csc_matrix(d1_l)
    d1_r = spsp.csc_matrix(d1_r)
    return H, HI, D1, D2, e_l, e_r, d1_l, d1_r


# Upwind 1D third order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_upwind_3rd(m,h,order)
def sbp_upwind_3rd(m, h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:4, 0:4] = np.diag(np.array(
        [0.4347899357e10 / 0.12695947216e11, 0.12032349023e11 / 0.9521960412e10, 0.32831414215e11 / 0.38087841648e11,
         0.6550489565e10 / 0.6347973608e10]))
    H[-4:, -4:] = np.fliplr(np.flipud(H[0:4, 0:4]))
    H = h * H

    HI = np.linalg.inv(H)

    Qp = -1 / 3 * np.diag(np.ones(m - 1), -1) - 1 / 2 * np.diag(np.ones(m), 0) + 1 * np.diag(np.ones(m - 1),
                                                                                             1) - 1 / 6 * np.diag(
        np.ones(m - 2), 2);

    Qu = np.array([
        [-0.847e3 / 0.37560e5, 0.79604458492699e14 / 0.119214944358240e15, -0.1643521867663e13 / 0.14901868044780e14,
         -0.4160444549287e13 / 0.119214944358240e15],
        [-0.22671019561497e14 / 0.39738314786080e14, -0.6023e4 / 0.37560e5, 0.91628011326497e14 / 0.119214944358240e15,
         -0.749671686919e12 / 0.19869157393040e14],
        [0.63495586071e11 / 0.1241822337065e13, -0.16644840223051e14 / 0.39738314786080e14, -0.4311e4 / 0.12520e5,
         0.104757273135509e15 / 0.119214944358240e15],
        [0.4998377065543e13 / 0.119214944358240e15, -0.5276507651527e13 / 0.59607472179120e14,
         -0.12476888349687e14 / 0.39738314786080e14, -0.5919e4 / 0.12520e5]])

    Qp[:4, :4] = Qu
    Qp[-4:, -4:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI @ (Qp - 0.5 * np.tensordot(e_l, e_l, axes=0) + 0.5 * np.tensordot(e_r, e_r, axes=0))
    Dm = HI @ (Qm - 0.5 * np.tensordot(e_l, e_l, axes=0) + 0.5 * np.tensordot(e_r, e_r, axes=0))

    return H, HI, Dp, Dm, e_l, e_r


# Upwind 1D fifth order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_upwind_5th(m,h,order)
def sbp_upwind_5th(m, h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:4, 0:4] = np.diag(np.array([0.251e3 / 0.720e3, 0.299e3 / 0.240e3, 0.211e3 / 0.240e3, 0.739e3 / 0.720e3]))
    H[-4:, -4:] = np.fliplr(np.flipud(H[0:4, 0:4]))
    H = h * H

    HI = np.linalg.inv(H)

    Qp = 1 / 20 * np.diag(np.ones(m - 2), -2) - 1 / 2 * np.diag(np.ones(m - 1), -1) - 1 / 3 * np.diag(np.ones(m),
                                                                                                      0) + np.diag(
        np.ones(m - 1), 1) - 1 / 4 * np.diag(np.ones(m - 2), 2) + 1 / 30 * np.diag(np.ones(m - 3), 3)

    Qu = np.array([
        [-0.1e1 / 0.120e3, 0.941e3 / 0.1440e4, -0.47e2 / 0.360e3, -0.7e1 / 0.480e3],
        [-0.869e3 / 0.1440e4, -0.11e2 / 0.120e3, 0.25e2 / 0.32e2, -0.43e2 / 0.360e3],
        [0.29e2 / 0.360e3, -0.17e2 / 0.32e2, -0.29e2 / 0.120e3, 0.1309e4 / 0.1440e4],
        [0.1e1 / 0.32e2, -0.11e2 / 0.360e3, -0.661e3 / 0.1440e4, -0.13e2 / 0.40e2]])

    Qp[:4, :4] = Qu
    Qp[-4:, -4:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI @ (Qp - 0.5 * np.tensordot(e_l, e_l, axes=0) + 0.5 * np.tensordot(e_r, e_r, axes=0))
    Dm = HI @ (Qm - 0.5 * np.tensordot(e_l, e_l, axes=0) + 0.5 * np.tensordot(e_r, e_r, axes=0))

    return H, HI, Dp, Dm, e_l, e_r


# Upwind 1D seventh order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_upwind_7th(m,h,order)
def sbp_upwind_7th(m, h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:6, 0:6] = np.diag(np.array(
        [0.19087e5 / 0.60480e5, 0.84199e5 / 0.60480e5, 0.18869e5 / 0.30240e5, 0.37621e5 / 0.30240e5,
         0.55031e5 / 0.60480e5, 0.61343e5 / 0.60480e5]))
    H[-6:, -6:] = np.fliplr(np.flipud(H[0:6, 0:6]))
    H = h * H

    HI = np.linalg.inv(H)

    Qp = -1 / 105 * np.diag(np.ones(m - 3), -3) + 1 / 10 * np.diag(np.ones(m - 2), -2) - 3 / 5 * np.diag(np.ones(m - 1),
                                                                                                         -1) - 1 / 4 * np.diag(
        np.ones(m), 0) + np.diag(np.ones(m - 1), 1) - 3 / 10 * np.diag(np.ones(m - 2), 2) + 1 / 15 * np.diag(
        np.ones(m - 3), 3) - 1 / 140 * np.diag(np.ones(m - 4), 4);

    Qu = np.array([
        [-0.265e3 / 0.300272e6, 0.1587945773e10 / 0.2432203200e10, -0.1926361e7 / 0.25737600e8,
         -0.84398989e8 / 0.810734400e9, 0.48781961e8 / 0.4864406400e10, 0.3429119e7 / 0.202683600e9],
        [-0.1570125773e10 / 0.2432203200e10, -0.26517e5 / 0.1501360e7, 0.240029831e9 / 0.486440640e9,
         0.202934303e9 / 0.972881280e9, 0.118207e6 / 0.13512240e8, -0.231357719e9 / 0.4864406400e10],
        [0.1626361e7 / 0.25737600e8, -0.206937767e9 / 0.486440640e9, -0.61067e5 / 0.750680e6,
         0.49602727e8 / 0.81073440e8, -0.43783933e8 / 0.194576256e9, 0.51815011e8 / 0.810734400e9],
        [0.91418989e8 / 0.810734400e9, -0.53314099e8 / 0.194576256e9, -0.33094279e8 / 0.81073440e8,
         -0.18269e5 / 0.107240e6, 0.440626231e9 / 0.486440640e9, -0.365711063e9 / 0.1621468800e10],
        [-0.62551961e8 / 0.4864406400e10, 0.799e3 / 0.35280e5, 0.82588241e8 / 0.972881280e9,
         -0.279245719e9 / 0.486440640e9, -0.346583e6 / 0.1501360e7, 0.2312302333e10 / 0.2432203200e10],
        [-0.3375119e7 / 0.202683600e9, 0.202087559e9 / 0.4864406400e10, -0.11297731e8 / 0.810734400e9,
         0.61008503e8 / 0.1621468800e10, -0.1360092253e10 / 0.2432203200e10, -0.10677e5 / 0.42896e5]
    ])

    Qp[:6, :6] = Qu
    Qp[-6:, -6:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI @ (Qp - 0.5 * np.tensordot(e_l, e_l, axes=0) + 0.5 * np.tensordot(e_r, e_r, axes=0))
    Dm = HI @ (Qm - 0.5 * np.tensordot(e_l, e_l, axes=0) + 0.5 * np.tensordot(e_r, e_r, axes=0))

    return H, HI, Dp, Dm, e_l, e_r


# Central 1D finite difference explicit and periodic operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   order - order of accuracy (2,4,6,8,10 or 12)
#   use_AD - if including artificial dissipation
# 
# Output:
#   H - inner product matrix 
#   Q - skew symmetric part of D1 = inv(H)*Q
#   D1- finite difference operator 
# 
# Use as follows:
# 
# import operators as ops
# H,Q,D1 = ops.periodic_expl(m,h,order,use_AD)
# 
def periodic_expl(m, h, order, use_AD=False):
    if order == 2:
        d = np.array([-0.5, 0, 0.5])
        l = 1
        r = 1
    elif order == 4:
        d = np.array([1. / 12, -2. / 3, 0, 2. / 3, -1. / 12])
        l = 2
        r = 2
    elif order == 6:
        d = np.array([-1. / 60, 3. / 20, -3. / 4, 0, 3. / 4, -3. / 20, 1. / 60])
        l = 3
        r = 3
    elif order == 8:
        d = np.array([1. / 280, -4. / 105, 1. / 5, -4. / 5, 0, 4. / 5, -1. / 5, 4. / 105, -1. / 280])
        l = 4
        r = 4
    elif order == 10:
        d = np.array(
            [-1. / 1260, 5. / 504, -5. / 84, 5. / 21, -5. / 6, 0, 5. / 6, -5. / 21, 5. / 84, -5. / 504, 1. / 1260])
        l = 5
        r = 5
    elif order == 12:
        d = np.array(
            [1. / 5544, -1. / 385, 1. / 56, -5. / 63, 15. / 56, -6. / 7, 0, 6. / 7, -15. / 56, 5. / 63, -1. / 56,
             1. / 385, -1. / 5544])
        l = 6
        r = 6
    else:
        raise NotImplementedError('Order not implemented.')

    v = np.zeros(m)
    for i in range(r + 1):
        v[i] = d[i + l]
    for i in range(l):
        v[m - i - 1] = d[l - i - 1]

    Q = spsp.csc_matrix(splg.toeplitz(np.roll(np.flip(v), 1), v))
    H = spsp.csc_matrix(h * np.eye(m))

    if use_AD:
        if order == 2:
            d = np.array([1, -2, 1])
            l = 1
            r = 1
            a = 0.5
        elif order == 4:
            d = -np.array([1, -4, 6, -4, 1])
            l = 2
            r = 2
            a = 1. / 12
        elif order == 6:
            d = np.array([1, -6, 15, -20, 15, -6, 1])
            l = 3
            r = 3
            a = 1. / 60
        elif order == 8:
            d = -np.array([1, -8, 28, -56, 70, -56, 28, -8, 1])
            l = 4
            r = 4
            a = 1. / 280
        elif order == 10:
            d = np.array([1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1])
            l = 5
            r = 5
            a = 1. / 1260
        elif order == 12:
            d = -np.array([1, -12, 66, -220, 495, -792, 924, -792, 495, -220, 66, -12, 1])
            l = 6
            r = 6
            a = 1. / 5544
        else:
            raise NotImplementedError('Order not implemented.')

        v = np.zeros(m)
        for i in range(r + 1):
            v[i] = d[i + l]
        for i in range(l):
            v[m - i - 1] = d[l - i - 1]

        S = spsp.csc_matrix(a * splg.toeplitz(np.roll(np.flip(v), 1), v))

        Q = Q - S

    D1 = 1 / h * Q

    return H, Q, D1


# Central 1D finite difference implicit and periodic operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   use_AD - if including artificial dissipation
# 
# Output:
#   H - inner product matrix 
#   Q - skew symmetric part of D1 = inv(H)*Q
# 
# Use as follows:
# 
# import operators as ops
# H,Q = ops.periodic_imp(m,h,use_AD)
# 
def periodic_imp(m, h, use_AD=False):
    h0 = 4203267613564094932432577824954. / 7049220443079284250976145948443;
    h1 = 22618790744689935699264926210401. / 84590645316951411011713751381316;
    h2 = -2209778222820418388602425303685. / 42295322658475705505856875690658;
    h3 = -1581945765. / 75409415044;
    h4 = 228992488. / 33235651987;
    h5 = 27214243. / 33751459947;

    q1 = 9607266784889201296177. / 19560081711822931675052;
    q2 = 8866705546306148289391. / 97800408559114658375260;
    q3 = -19659090145677941034997. / 293401225677343975125780;
    q4 = 127051314. / 37983174851;
    q5 = 389910724. / 128741750713;

    # Q
    l = 5
    r = 5
    d = np.array([-q5, -q4, -q3, -q2, -q1, 0, q1, q2, q3, q4, q5])

    v = np.zeros(m)
    for i in range(r + 1):
        v[i] = d[i + l]
    for i in range(l):
        v[m - i - 1] = d[l - i - 1]

    Q = spsp.csc_matrix(splg.toeplitz(np.roll(np.flip(v), 1), v))

    # H
    l = 5
    r = 5
    d = np.array([h5, h4, h3, h2, h1, h0, h1, h2, h3, h4, h5])

    v = np.zeros(m)
    for i in range(r + 1):
        v[i] = d[i + l]
    for i in range(l):
        v[m - i - 1] = d[l - i - 1]

    H = spsp.csc_matrix(h * splg.toeplitz(np.roll(np.flip(v), 1), v))

    if use_AD:
        d = -np.array([1, -12, 66, -220, 495, -792, 924, -792, 495, -220, 66, -12, 1])
        l = 6
        r = 6
        a = 1. / 5544

        v = np.zeros(m)
        for i in range(r + 1):
            v[i] = d[i + l]
        for i in range(l):
            v[m - i - 1] = d[l - i - 1]

        S = spsp.csc_matrix(a * splg.toeplitz(np.roll(np.flip(v), 1), v))

        Q = Q - S

    return H, Q


# Central 1D finite difference variable coefficient second derivative periodic operator.
# Constructed from the corresponding D1 operators as follows: D2(c) = D1*diag(c)*D1.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   order - order of accuracy (2,4,6,8,10 or 12)
# 
# Output:
#   M_fun - function computing the variable coefficient matrix M
# 
# Use as follows:
# 
# import operators as ops
# M_fun = ops.periodic_variable_wide(m,h,order)
# 
def periodic_variable_wide(m, h, order):
    H, Q, _ = periodic_expl(m, h, order, False)

    def M_fun(c):
        C = np.diag(c)
        M = -1 / h * Q @ C @ Q
        return M

    return M_fun
