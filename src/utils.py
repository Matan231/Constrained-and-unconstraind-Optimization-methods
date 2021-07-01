import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # appropriate import to draw 3d polygons
from matplotlib import style

def report(iter_num, x_prev, x_next, f_prev, f_next,success):
    print(f"Itaration: {iter_num} current location: {x_next} obj value: {f_next} step length: {np.linalg.norm(x_next - x_prev)} change in Value: {np.linalg.norm(f_next-f_prev)}, {success}")



def plotting(f, path, plot_range = 3.0, num_of_counter_lines = 55 , title = "no title has been set"):

    delta = 0.025
    x = np.arange(-plot_range, plot_range, delta)
    y = np.arange(-plot_range, plot_range, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))[0]

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels = num_of_counter_lines)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title)

    for i in range(0, len(path)-1):
        arrow = FancyArrowPatch((path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]), arrowstyle='simple', color='k', mutation_scale=10)
        ax.add_patch(arrow)
        #plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]])

    plt.show()

    plot_val_iter(f, path, title)

def plot_val_iter(f, path, title):
    obj_values = [f(i)[0] for i in path]
    plt.plot([i for i in range(len(obj_values))], obj_values, marker='o')
    plt.xlabel("Number of iterations")
    plt.ylabel("Objective function value")
    plt.title(title)
    #plt.yscale('log')
    plt.show()


def plot_feasibile_lp(f, path, plot_range = 3.0, num_of_counter_lines = 25 , title = "no title has been set"):

    delta = 0.025
    x = np.arange(-plot_range, plot_range, delta)
    y = np.arange(-plot_range, plot_range, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))[0]

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=num_of_counter_lines)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title)

    for i in range(0, len(path) - 1):
        arrow = FancyArrowPatch((path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1]), arrowstyle='simple',
                                color='k', mutation_scale=10)
        ax.add_patch(arrow)
        # plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]])

    # plot the feasible region
    d = np.linspace(-2, 16, 300)
    x, y = np.meshgrid(d, d)
    plt.imshow(((y >= -x + 1) & (y <= 1) & (2 >= x) & (0 <= y)).astype(int),
               extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3);

    # plot the lines defining the constraints
    x = np.linspace(-10, 16, 2000)
    # y >= -x+1
    y1 = (-x) + 1
    # y <= 1
    y2 = (x * 0) + 1
    # 4y >= 2x - 8
    y3 = x * 0 + 2
    # y <= 2x - 5
    y4 = x * 0

    # Make plot
    plt.plot(x, y1)
    plt.plot(y3, x)
    plt.plot(x, y2)
    plt.plot(x, y4)
    plt.xlim(-plot_range, plot_range)
    plt.ylim(-plot_range, plot_range)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    print(path[-1])
    plt.plot(path[-1][0], path[-1][1] , 'ro')

    plt.show()

def plot_feasibile_qp(func, path):
    fig =plt.figure()
    custom = fig.add_subplot(111, projection='3d')

    x1 = np.array([1, 0, 0])
    y1 = np.array([0, 1, 0])
    z1 = np.array([0, 0, 1])
    custom.scatter(x1, y1, z1)

    verts = [list(zip(x1, y1, z1))]
    srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
    plt.gca().add_collection3d(srf)

    custom.set_xlabel('X')
    custom.set_ylabel('Y')
    custom.set_zlabel('Z')


    points = np.stack( path, axis=0 ).T


    custom.plot(points[0], points[1], points[2], marker='x')
    custom.scatter(*points.T[-1], color='red')

    plt.show()

