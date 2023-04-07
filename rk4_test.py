import numpy as np
import matplotlib.pyplot as plt
import rungekutta4 as rk4

if __name__ == '__main__':
    w = 3 * 2 * np.pi
    amp = 0.1


    def g(t, _): return amp * w * np.sin(w * t)


    ht = 0.003
    u0 = np.array([0])
    ts = rk4.RK4Timestepper(1.9, ht, g, u0, store_v=True, N=1)
    ts.run_sim()
    t = ts.t_vec

    plt.plot(t, ts.vl.reshape((ts.mt+1,)), label='rk4')
    t = np.linspace(0, 1.9, 500)
    plt.plot(t, amp * (1-np.cos(w * t)), label='analytic')
    plt.legend()
    plt.show()

