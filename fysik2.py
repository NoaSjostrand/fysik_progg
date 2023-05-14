import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def calculate_z_vector(t, z):
    x, y, vx, vy = z
    z_vector = np.array([
        vx,
        vy,
        -x / (x**2 + y**2)**(3/2),
        -y / (x**2 + y**2)**(3/2)
    ])
    return z_vector


def calculate_values(t, z):
    x, y, vx, vy = z
    L = x * vy - y * vx
    K = 0.5 * (vx**2 + vy**2)
    U = -1 / np.sqrt(x**2 + y**2)
    E = K + U
    return L, K, U, E

# Initial conditions
x0, y0 = 1, 0
vx0, vy0 = 0, 1.2

z0 = np.array([x0, y0, vx0, vy0])
t0, tf = 0, 30
increments = 100000
t = np.linspace(t0, tf, increments)

sol = solve_ivp(calculate_z_vector, [t[0], t[-1]], z0, dense_output=True, max_step=0.01)

# Calculate values
L_values, K_values, U_values, E_values = calculate_values(sol.t, sol.y)


# ----------------------------------------


# Plot
plt.figure(figsize=(8, 6))

# Position
plt.subplot(2, 2, 1)
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Position')
plt.grid(True)


# Angular Momentum
plt.subplot(2, 2, 2)
plt.plot(sol.t, L_values)
plt.autoscale(enable=True, axis='y')
plt.xlabel('t')
plt.ylabel('L(t)')
plt.title('Angular Momentum')

# Energies
plt.subplot(2, 2, 3)
plt.plot(sol.t, K_values)
plt.plot(sol.t, U_values)
plt.plot(sol.t, E_values)
plt.legend(['Kinetic', 'Potential', 'Total'])
plt.xlabel('t')
plt.title('Energies')

plt.tight_layout()
plt.show()
