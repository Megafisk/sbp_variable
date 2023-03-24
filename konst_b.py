import numpy as np
from D2_Variable import *
import scipy.sparse as spsp
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
m = 101
N = m * m
xvec, h = np.linspace(0, 1, m, retstep=True)
yvec = np.linspace(0, 1, m)
X, Y = np.meshgrid(xvec, yvec, indexing='ij')
x = X.reshape((N, 1))
y = Y.reshape((N, 1))
y_left = yvec.reshape((m, 1))

# define wave speeds
a = 1
b = 1
c = (a * b) ** 0.5
B = np.ones(N) * b  # constant wave speed

# initial data
# v = np.zeros((N, 1))
sigma = 0.05
x0 = 0.5
y0 = 0.5
v = np.exp(-(x - x0) ** 2 / sigma ** 2 - (y - y0) ** 2 / sigma ** 2)
v_t = np.zeros((N, 1))
u = np.vstack((v, v_t))

# time stuff
T = 2
ht = 0.1 * h / c
mt = int(math.ceil(T / ht) + 1)
tvec, ht = np.linspace(0, T, mt, retstep=True)

# gaussian inflow data
# t0 = 1
# def g(t): return np.exp(-(t - t0 - 2 * y_left) ** 2) * np.ones((m, 1))

# define operators
# ops_1d = D2_Variable_4(m, h)
# H, HI, D1, D2_fun, e_l, e_r, d1_l, d1_r = ops_1d
# HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = d2_2d_variable_4(m, h, B, ops_1d)

# operators with constant b
op = ops.sbp_cent_4th(m, h)
H, HI, D1, D2, e_l, e_r, d1_l, d1_r = op
HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = ops.convert_1d_2d(op, m)

# D = a * (D2x + D2y) - HHI @ (eW @ d1_W.T + eE @ d1_E.T + eN @ d1_N.T + eS @ d1_S.T)
# C = - HHI @ eE @ eE.T
# G = - HHI @ eW
D = D2x + D2y - HHI @ (-eW @ H @ d1_W.T + eE @ H @ d1_E.T + eN @ H @ d1_N.T - eS @ H @ d1_S.T)


DD = spsp.bmat(((None, spsp.eye(N)), (D, None)))
zeros_N = np.zeros((N, 1))


# plot
zlow = 0
zhigh = 0.5
ax: plt.Axes
fig: plt.Figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
plt.xlabel("x")
plt.ylabel("y")
# mesh = ax.pcolormesh(X, Y, v.reshape((m, m)), shading="nearest", vmin=zlow, vmax=zhigh)
mesh = ax.imshow(v.reshape(m, m), vmin=zlow, vmax=zhigh)
title = plt.title("Phi at t = " + str(0))
fig.colorbar(mesh, ax=ax)
fig.tight_layout()
plt.draw()
plt.pause(0.5)


def rhs(u): return DD @ u
# def gg(t): return np.vstack((zeros_N, G @ g(t)))
def gg(t): return 0


t = 0
for t_i in range(mt - 1):
    # Take one step with the fourth order Runge-Kutta method.
    u, t = rk4.step(rhs, u, gg, t, ht)

    # Update plot every 20th time step
    if (t_i % 20) == 0 or t_i == mt - 2:
        v = u[0:N]
        # phi_t = v[mtot:2 * mtot]

        mesh.set_array(v.reshape(m, m))
        # print(tvec[t_i + 1])

        title.set_text(f"t = {tvec[t_i + 1]:.2f}")
        plt.draw()
        plt.pause(0.05)

plt.show()
