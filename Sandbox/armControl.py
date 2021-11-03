from math import acos, pow, atan, sin, pi, cos

import matplotlib.pyplot as plt

x = 325
y = 101
L1 = 258
L2 = 151
theta2 = acos((pow(x, 2) + pow(y, 2) - pow(L1, 2) - pow(L2, 2)) / (2 * L1 * L2))
if x < 0 and y > 0:
    theta2 *= -1
# else:
#     theta2 *= -1
#     theta2 -= pi
theta1 = atan(x / y) - atan((L2 * sin(theta2)) / (L1 + L2 * cos(theta2)))
theta2 = -1 * theta2 * 180 / pi
theta1 = theta1 * 180 / pi
if x >= 0 and y >= 0:
    theta11 = 90 - theta1
elif x < 0 and y > 0:
    theta11 = 90 - theta1
if x > 0:
    theta2 = theta2 * -1 + 180
else:
    theta2 = 180 - theta2
y1 = L1 * sin(theta11 * pi / 180)
x1 = L1 * cos(theta11 * pi / 180)
print("L1: " + str(theta11))
print("L2: " + str(theta2))
plt.plot([-325, -325, 325, 325, -325], [101, 401, 401, 101, 101])
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
