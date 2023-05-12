import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-10,10,1)
y = np.arange(-10,10,1)

# Creating the meshgrid 
X, Y = np.meshgrid(x,y)

#Creating delx and dely array
delx = np.zeros_like(X)
dely = np.zeros_like(Y)
s = 7
r=2

for i in range(len(x)):
  for j in range(len(y)):
    
    # calculating the distance asumming the goal is at Origin
    d= np.sqrt(X[i][j]**2 + Y[i][j]**2)
    #print(f"{i} and {j}")

    # calCulating the Theta
    theta = np.arctan2(Y[i][j],X[i][j])

    # Using the equations given in the class
    if d< 2:
      delx[i][j] = 0
      dely[i][j] =0
    elif d>r+s:
      delx[i][j] = -50* s *np.cos(theta)
      dely[i][j] = -50 * s *np.sin(theta)
    else:
      delx[i][j] = -50 * (d-r) *np.cos(theta)
      dely[i][j] = -50 * (d-r) *np.sin(theta)
