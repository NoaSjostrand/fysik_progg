import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

x_0, y_0 = 1, 0
vx0, vy0 = 0, 0.5
t0 = 0
t1 = 10

saved_x = []
saved_y = []
saved_vx = []
saved_vy = []

k = 1
m = 1

#   Initiala värden
z_0 = np.array([x_0, y_0, vx0, vy0])


#   Kraftfält i en punkt
def kraft_i_punkt(x, y):
    return -k * (x / ((x ** 2 + y ** 2) ** (3 / 2))), -k * (y / ((x ** 2 + y ** 2) ** (3 / 2)))


def update_particle_position():
    x, y, vx, vy = z_0
    t_step = 0.001
    for t in np.arange(t0, t1, t_step):

        fx, fy = kraft_i_punkt(x, y)
        ax, ay = fx/m, fy/m

        vx += ax*t
        vy += ay*t
        x += vx*t
        y += vy*t

        saved_x.append(x)
        saved_y.append(y)
        saved_vx.append(vx)
        saved_vy.append(vy)

update_particle_position()
plt.scatter(saved_x, saved_y)
plt.show()

