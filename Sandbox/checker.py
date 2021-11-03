from math import acos, pi, atan, sin, cos

import numpy as np

for x in np.arange(280,284,0.5):
    for y in np.arange(90, 120, 0.5):
        for L1 in np.arange(257, 259,0.5):
            for L2 in np.arange(151, 153, 0.5):
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
                    if 266 < theta2 < 267 and 52 < theta11 < 53:
                        # x = x * -1
                        # theta2 = acos((pow(x, 2) + pow(y, 2) - pow(L1, 2) - pow(L2, 2)) / (2 * L1 * L2))
                        # if x < 0 and y > 0:
                        #     theta2 *= -1
                        # # else:
                        # #     theta2 *= -1
                        # #     theta2 -= pi
                        # theta1 = atan(x / y) - atan((L2 * sin(theta2)) / (L1 + L2 * cos(theta2)))
                        # theta2 = -1 * theta2 * 180 / pi
                        # theta1 = theta1 * 180 / pi
                        # if x >= 0 and y >= 0:
                        #     theta11 = 90 - theta1
                        # elif x < 0 and y > 0:
                        #     theta11 = 90 - theta1
                        # if x > 0:
                        #     theta2 = theta2 * -1 + 180
                        # else:
                        #     theta2 = 180 - theta2
                        # if 88 < theta2 < 92 and 124 < theta11 < 128:
                            print(x)
                            print(y)
                            print(L1)
                            print(L2)
                            print()
                except:
                    continue
