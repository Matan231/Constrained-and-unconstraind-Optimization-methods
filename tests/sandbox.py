
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

points = np.array([[0.86063431, 0.07119279, 1.70377142],
                   [0.86391084, 0.07014899, 1.72184785],
                   [0.86332177, 0.069444, 1.71182579],
                   [0.86192988, 0.06913941, 1.69818289],
                   [0.86166436, 0.06916367, 1.69527615]]).T
path = [np.ones(3), np.zeros(3), np.ones(3)]
arr = np.stack( path, axis=0 ).T
print(arr)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[0], points[1], points[2], marker='x')
#ax.scatter(*points.T[0], color='red')
#plt.show()