from math import acos, pow, atan, sin, pi, cos

import matplotlib.pyplot as plt

x = 366 - 50
y = 100 + 50
L1 = 300
L2 = 200
theta2 = acos((pow(x, 2) + pow(y, 2) - pow(L1, 2) - pow(L2, 2)) / (2 * L1 * L2))
if x < 0 and y < 0:
    theta2 = -1 * theta2
theta1 = atan(x / y) - atan((L2 * sin(theta2)) / (L1 + L2 * cos(theta2)))
theta2 = -1 * theta2 * 180 / pi
theta1 = theta1 * 180 / pi
if x >= 0 and y >= 0:
    theta11 = 90 - theta1
elif x < 0 and y > 0:
    theta11 = 90 - theta1
y1 = L1 * sin(theta11 * pi / 180)
x1 = L1 * cos(theta11 * pi / 180)

print("L1: " + str(theta11))
print("L2: " + str(theta2))
plt.plot([-366, -366, 366, 366, -366], [100, 400, 400, 100, 100])
plt.plot([0, x1, x], [0, y1, y], "r")
plt.grid()
plt.ylim(-600, 600)
plt.xlim(-600, 600)
plt.show()

phi = theta1 + theta2
phi = (-1) * phi
if abs(phi) > 165:
    phi = 180 + phi
print(phi)
