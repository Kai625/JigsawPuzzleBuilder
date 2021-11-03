from math import acos, pi, atan, sin, cos

for x in range(1, 326):
    for y in range(0, 130):
        for L1 in range(257, 259):
            for L2 in range(150, 154):
                try:
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
                    if 250 < theta2 < 252 and 41 < theta11 < 42:
                        print(x)
                        print(y)
                        print(L1)
                        print(L2)
                        print()
                except:
                    continue
