import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

x_0, y_0 = [1, 0]
vx0, vy0 = [0, 0]
t0 = 0
t1 = 10

z_0 = [x_0, y_0, vx0, vy0]
