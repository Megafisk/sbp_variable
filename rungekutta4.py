from typing import Union, Sequence

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
    def __init__(self, T, htt, rhs, u0, update=lambda ts: None,
                 save_every: Union[int, float, Sequence, np.ndarray] = -1, N=None,
                 before_step=lambda ts: None):
        self.mt = int(np.ceil(T / htt))  # number of timesteps to take
        # mt+1 since the initial value is already given
        self.t_vec, self.ht = np.linspace(0, T, self.mt + 1, retstep=True)
        self.T = T
        self.f = rhs
        self.t = 0
        self.t_i = 0
        self.u = u0
        self.update = update
        self.before_step = before_step
        if N:
            self.N = N
        else:
            self.N = len(u0) // 2

        if isinstance(save_every, float):
            save_every = np.union1d(np.searchsorted(self.t_vec, np.arange(0, self.T, save_every)), [self.mt])
        self._save_specified = not isinstance(save_every, int)
        self._do_save = self._save_specified or save_every > 0
        if self._do_save:
            self._save_i = 0
            if self._save_specified:
                self.saved_frames = save_every.ravel() if isinstance(save_every, np.ndarray) else save_every
                self.vl = np.zeros((self.N, len(self.saved_frames)))
            else:
                self.vl = np.zeros((self.N, np.ceil(self.mt / save_every + 1).astype('int')))
                self.saved_frames = np.zeros((self.vl.shape[1],), dtype='int')
                self.save_every = save_every

            if 0 in self.saved_frames:
                self.vl[:self.N, 0] = self.v().reshape((self.N,))
                self._save_i = 1

    def step(self):
        self.before_step(self)
        self.u, self.t = step(self.f, self.u, self.t, self.ht)
        self.t_i += 1
        self.update(self)

    def run_sim(self):
        while self.t_i < self.mt:
            self.step()
            if self._do_save:
                if (self.t_i in self.saved_frames
                        or not self._save_specified and (self.t_i % self.save_every == 0 or self.t_i == self.mt)):
                    self.vl[:self.N, self._save_i] = self.v().reshape((self.N,))
                    self.saved_frames[self._save_i] = self.t_i
                    self._save_i += 1
        return self

    def v(self):
        return self.u[:self.N]

    def vt(self):
        return self.u[self.N:]
