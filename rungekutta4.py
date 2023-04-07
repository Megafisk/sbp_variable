import numpy as np


def step(f, v, t, dt):
    """Take one RK4 step. Return updated solution and time.
    f: Right-hand-side function: dv/dt = f(t, v)
    v: current solution
    t: current time
    dt: time step

    @return: v, t
    """

    # Compute rates k1-k4
    k1 = dt * (f(t, v))
    k2 = dt * (f(t + 0.5 * dt, v + 0.5 * k1))
    k3 = dt * (f(t + 0.5 * dt, v + 0.5 * k2))
    k4 = dt * (f(t + dt, v + k3))

    # Update solution and time
    v = v + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t = t + dt

    return v, t


class RK4Timestepper:
    def __init__(self, T, htt, rhs, u0, update=lambda ts: None, store_v=False, N=None):
        self.mt = int(np.ceil(T / htt))  # number of timesteps to take
        self.t_vec, self.ht = np.linspace(0, T, self.mt+1, retstep=True)  # mt+1 since the inital value is already given
        self.T = T
        self.f = rhs
        self.t = 0
        self.t_i = 0
        self.u = u0
        self.update = update
        if N:
            self.N = N
        else:
            self.N = len(u0) // 2
        self.store_v = store_v
        if store_v:
            self.vl = np.zeros((self.N, len(self.t_vec)))
            self.vl[:self.N, 0] = self.v().reshape((self.N,))

    def step(self):
        self.u, self.t = step(self.f, self.u, self.t, self.ht)
        self.t_i += 1
        self.update(self)

    def run_sim(self):
        while self.t_i < self.mt:
            self.step()
            if self.store_v:
                self.vl[:self.N, self.t_i] = self.v().reshape((self.N,))

    def v(self):
        return self.u[:self.N]

    def vt(self):
        return self.u[self.N:]
