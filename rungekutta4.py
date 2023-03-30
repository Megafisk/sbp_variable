"""4th order Runge-Kutta time-stepping.
This module solves ODEs of the form dv/dt = f(t, v)
"""


def step(f, v, t, dt):
    """Take one RK4 step. Return updated solution and time.
    f: Right-hand-side function: dv/dt = f(t, v)
    v: current solution
    t: current time
    dt: time step
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
