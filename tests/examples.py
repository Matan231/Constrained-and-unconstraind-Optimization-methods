import numpy as np
from src.unconstrained_min import line_search


def f_1(x, hessian = False):
    Q = np.array([[1, 0], [0, 1]])
    f_x = np.matmul(x, np.matmul(Q, x))
    df_x = np.array(2*x)
    if hessian:
        H = Q * 2
        return f_x, df_x, H
    return f_x, df_x


def f_2(x, hessian = False):
    Q = np.array([[5, 0], [0, 1]])
    f_x = np.matmul(x, np.matmul(Q, x))
    df_x = np.matmul(Q, x)*2
    if hessian:
        H = Q*2
        return f_x, df_x, H
    return f_x, df_x


def f_3(x, hessian = False):
    sqr_3 = np.sqrt(3)
    b_matrix = np.array([[5, 0], [0, 1]])
    a_matrix = np.array([[sqr_3/2, -0.5], [0.5, sqr_3/2]])
    Q = np.matmul(np.matmul(a_matrix.transpose(), b_matrix), a_matrix)
    f_x = np.matmul(x, np.matmul(Q, x))
    df_x = np.matmul(Q+np.transpose(Q), x)
    if hessian:
        H = Q + np.transpose(Q)
        return f_x, df_x, H
    return f_x, df_x


def Rosenbrock(x, hessian = False):
    f_x = 100*((x[1] - x[0]**2)**2) + (1 - x[0])**2
    pd_f_x1 = -400*(x[1] -x[0]**2)*x[0] - 2*(1 - x[0])
    pd_f_x2 = 200*(x[1] -x[0]**2)
    df_x = np.array([pd_f_x1, pd_f_x2])
    if hessian:
        H = np.array([[1200*x[0]**2-400*x[1] + 2, -400*x[0]],
                      [-400*x[0], 200]])
        return f_x, df_x, H
    return f_x, df_x


def f_linear(x, hessian = False):
    a = [2, 3]
    f_x = np.dot(a, x)
    df_x = np.array(a)
    if hessian:
        pass
    return f_x, df_x

def min_obj_lp(x):
    f_x = x[0] + x[1]
    df_x = np.array([1, 1])
    hessian_x = np.zeros((2, 2))
    return f_x, df_x, hessian_x

def obj_lp(x):
    f_x = -x[0] -x[1]
    df_x = np.array([-1, -1])
    hessian_x = np.zeros((2, 2))
    return f_x, df_x, hessian_x

def lp_ineq_1(x):
    f_x = -x[0]-x[1] + 1
    df_x = np.array([-1, -1])
    hessian_x = np.zeros((2, 2))
    return f_x,  df_x, hessian_x

def lp_ineq_2(x):
    f_x = x[1] - 1
    df_x = np.array([0, 1])
    hessian_x = np.zeros((2, 2))
    return f_x,  df_x, hessian_x

def lp_ineq_3(x):
    f_x = x[0] - 2
    df_x = np.array([1, 0])
    hessian_x = np.zeros((2, 2))
    return f_x,  df_x, hessian_x

def lp_ineq_4(x):
    f_x = -x[1]
    df_x = np.array([0, -1])
    hessian_x = np.zeros((2, 2))
    return f_x,  df_x, hessian_x

def obj_qp(x):
    fx = x[0]**2 + x[1]**2 + (x[2]+1)**2
    dfx = np.array([2*x[0], 2*x[1], 2*(x[2]+1)])
    hessian = np.array([[2,0,0],[0,2,0],[0,0,2]])
    return fx, dfx, hessian


def qp_ineq_1(x):
    fx = -x[0]
    dfx = np.array([-1,0,0])
    hessian = np.zeros((3,3))
    return fx, dfx, hessian


def qp_ineq_2(x):
    fx = -x[1]
    dfx = np.array([0, -1, 0])
    hessian = np.zeros((3, 3))
    return fx, dfx, hessian


def qp_ineq_3(x):
    fx = -x[2]
    dfx = np.array([0, 0, -1])
    hessian = np.zeros((3, 3))
    return fx, dfx, hessian

