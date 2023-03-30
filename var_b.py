import numpy as np
from D2_Variable import *
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplg
import math
from matplotlib import pyplot as plt

import rungekutta4 as rk4
import operators as ops

# define grid
# x_min = 0
# x_max = 1
# y_min = 0
# y_max = 1
# mx = 41
# my = 61
mb = 20
T = 10

m = 3 * mb
N = m * m
xvec, h = np.linspace(0, 1, m, retstep=True)
yvec = np.linspace(0, 1, m)
X, Y = np.meshgrid(xvec, yvec, indexing='ij')
x = X.reshape((N, 1))
y = Y.reshape((N, 1))
y_left = yvec.reshape((m, 1))

# define wave speeds
a0 = 1
a1 = 0.1
b0 = 1
b1 = 0.1
A = np.ones((m, m)) * a0
B = np.ones((m, m)) * b0
# B[mb:2*mb, mb:2*mb] = b1  # block of different wave speeds
# A[mb:2*mb, mb:2*mb] = a1
# B[(X - Y < 1 / 3) & (X - Y > 0) & (1 / 3 < X) & (X < 2 / 3) & (1 / 3 < Y) & (Y < 2 / 3)] = b1
B[(Y > 1/3) & (1 / 3 < X) & (X < 2 / 3)] = b1  # SKAPAR INSTABIL!
a = A.reshape((N,))
b = B.reshape((N,))

zlow = -0.5
zhigh = 0.5

# initial data
sigma = 0.05
x0 = 0.6
y0 = 0.1
v = np.zeros((N, 1))
# v = np.exp(-(x - x0) ** 2 / sigma ** 2 - (y - y0) ** 2 / sigma ** 2)
v_t = np.zeros((N, 1))
u = np.vstack((v, v_t))

# gaussian inflow data
t0 = 0.25
w = 0.1
amp = 0.4
def g(t): return amp * -2 * (t - t0) / (w ** 2) * np.exp(-(t - t0 + 0 * (y_left - 1)) ** 2 / (w ** 2))
# def g(t): return np.zeros((m, 1))
# wave inflow data
# freq = 3
# amp = 0.1
# omega = 2 * np.pi * freq
# def g(t): return amp * omega * np.sin(freq * 2 * np.pi * np.ones((m, 1)) * t)


# define operators
print('building operators...')

# operators with constant b
# op = ops.sbp_cent_4th(m, h)
# H, HI, D1, D2, e_l, e_r, d1_l, d1_r = op
# HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = ops.convert_1d_2d(op, m)

ops_1d = D2_Variable_4(m, h)
H, HI, D1, D2_fun, e_l, e_r, d1_l, d1_r = ops_1d
HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = d2_2d_variable_4(m, b, ops_1d)
print('operators done!')

AA = spsp.diags(a)
BB = spsp.diags(b)

# D = a * (D2x + D2y) - HHI @ (eW @ d1_W.T + eE @ d1_E.T + eN @ d1_N.T + eS @ d1_S.T)
C = - HHI @ eE @ H @ eE.T
G = HHI @ eW @ H
D = (D2x + D2y) - HHI @ (- eW @ H @ spsp.diags(B[0, :]) @ d1_W.T + eE @ H @ spsp.diags(B[-1, :]) @ d1_E.T
                         + eN @ H @ spsp.diags(B[:, -1]) @ d1_N.T - eS @ H @ spsp.diags(B[:, 0]) @ d1_S.T)

DD = spsp.bmat(((None, spsp.eye(N)), (D, C)))
zeros_N = np.zeros((N, 1))

# time stuff
ht = 0.5 * 2.8 / np.sqrt(abs(spsplg.eigs(D, 1)[0][0]))
mt = int(math.ceil(T / ht) + 1)
tvec, ht = np.linspace(0, T, mt, retstep=True)

# plot stuff
ax: plt.Axes
fig: plt.Figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
plt.xlabel("x")
plt.ylabel("y")
mesh = ax.imshow(v.reshape((m, m), order='F'), vmin=zlow, vmax=zhigh, origin='lower')
title = plt.title("Phi at t = " + str(0))
fig.colorbar(mesh, ax=ax)
fig.tight_layout()
plt.draw()
plt.pause(0.5)


def rhs(t, u): return DD @ u + np.vstack((zeros_N, G @ g(t)))


t = 0
for t_i in range(mt - 1):
    # Take one step with the fourth order Runge-Kutta method.
    u, t = rk4.step(rhs, u, t, ht)

    # Update plot every 20th time step
    if (t_i % 5) == 0 or t_i == mt - 2:
        v = u[0:N]

        mesh.set_array(v.reshape((m, m), order='F'))
        # print(tvec[t_i + 1])
        # print((G @ g(t))[0])

        title.set_text(f"t = {tvec[t_i + 1]:.2f}")
        # print(energy(u))
        # plt.draw()
        plt.pause(0.01)

plt.show()
