from math import ceil

import matplotlib.collections
from matplotlib import pyplot as plt, animation
import numpy as np
from scipy.sparse import kron, eye, bmat

import operators as ops
import rungekutta4 as rk4

mx = 200
my = 100
mtot = (mx+1) * (my+1)
hx = 2 / mx
hy = 1 / my
x_min = -1
x_max = 1
y_min = -1 / 2
y_max = 1 / 2
c = 1


x = np.linspace(x_min, x_max, mx+1)
y = np.linspace(y_min, y_max, my+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# initial data
x0 = 0
y0 = 0
sigma = 0.05
phi = np.exp(-(X - x0) ** 2 / sigma ** 2 - (Y - y0) ** 2 / sigma ** 2)
phi_t = np.zeros((mx+1) * (my+1))

# other initial conditions for faffing about
# kx = 0*np.pi
# ky = 2*2*np.pi
# w = c*np.sqrt(kx**2 + ky**2)
# phi = np.cos(kx*X) * np.cos(ky*Y)
# phi = 1-X**4-Y**2 + np.sqrt(X**2 + Y**2)

phi = phi.reshape(mtot, order="F")
v = np.hstack((phi, phi_t))


zlow = 0
zhigh = 0.2

# RK4-stuff
ht = 0.08 * hx / c
T = 15
mt = int(ceil(T / ht) + 1)
tvec, ht = np.linspace(0, T, mt, retstep=True)

# making the matrices
H_x, HI_x, D1_x, D2_x, e_l_x, e_r_x, d1_l_x, d1_r_x = ops.sbp_cent_4th(mx+1, hx)
H_y, HI_y, D1_y, D2_y, e_l_y, e_r_y, d1_l_y, d1_r_y = ops.sbp_cent_4th(my+1, hy)
D_mx = c ** 2 * (D2_x + HI_x * (e_l_x * d1_l_x.T - e_r_x * d1_r_x.T))
D_my = c ** 2 * (D2_y + HI_y * (e_l_y * d1_l_y.T - e_r_y * d1_r_y.T))

I_mx = eye(mx+1)
I_my = eye(my+1)
D = kron(I_my, D_mx) + kron(D_my, I_mx)

C = bmat(((None, eye(mtot)), (D, None)))


def rhs(v): return C @ v
def g(t): return 0


# plot
ax: plt.Axes
fig: plt.Figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
plt.xlabel("x")
plt.ylabel("y")
mesh: matplotlib.collections.QuadMesh
mesh = ax.pcolormesh(X, Y, phi.reshape((mx+1, my+1), order="F"), shading="nearest", vmin=zlow, vmax=zhigh)
title = plt.title("Phi at t = " + str(0))
fig.colorbar(mesh, ax=ax)
fig.tight_layout()
plt.draw()
plt.pause(0.5)

frames = [phi.reshape((mx+1, my+1), order="F")]

# Runge-Kutta 4
t = 0
for t_i in range(mt - 1):
    # Take one step with the fourth order Runge-Kutta method.
    v, t = rk4.step(rhs, v, g, t, ht)

    # Update plot every 20th time step
    if (t_i % 10) == 0 or t_i == mt - 2:
        phi = v[0:mtot]
        # phi_t = v[mtot:2 * mtot]

        data = phi.reshape((mx + 1, my + 1), order='F')
        mesh.set_array(data)
        frames.append(data)
        # print(tvec[t_i + 1])

        title.set_text(f"t = {tvec[t_i + 1]:.2f}")
        plt.draw()
        plt.pause(1e-3)


def animate(i):
    mesh.set_array(frames[i])
    return mesh


# anim = animation.FuncAnimation(fig, animate, interval=40, frames=800)
# anim.save('/Users/hermanbergkvist/Unigrejs/Bervet PDE/ut2.gif')


plt.show()
