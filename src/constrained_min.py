from functools import partial
from src.unconstrained_min import line_search
import numpy as np
from src.unconstrained_min import wolf_condition
from src.unconstrained_min import report
from src.unconstrained_min import check_converge
import numdifftools as nd


def interior_pt(func, x0, ineq_constraints, eq_constraints_mat= np.array([None]), eq_constraints_rhs=np.array([None])):

    total_path = []
    u = 2
    t = 1
    e = 0.01
    x_t = x0

    if len(ineq_constraints) > 0:
        func = partial(aug_func, func, ineq_constraints)
    while True:


        if eq_constraints_mat.all() == None:
            max_iterations = 100
            step_tol = 10 ** (-12)
            obj_tol = 10 ** (-12)
            x_t, suc, path = newton_dir(func, x_t, obj_tol, step_tol, max_iterations, init_step_len=1.0, slope_ratio=np.e**(-4), back_track_factor=0.2, t=t)
        else:
            x_t, suc, path = const_nt_decent(func, x_t, eq_constraints_mat, eq_constraints_rhs, t,e)
        if term_cond(ineq_constraints, t, e):
            total_path += path
            break

        t = u*t
        total_path += path



    return x_t, total_path,True


def aug_func(f, ineq_constraints, x, t=1, hessian=True):

    f_x, df_x, hessian_f_x = f(x)
    ineq_g = []
    ineq_dg = []
    ineq_hes_g = []

    for g in ineq_constraints:
        gx, dg, hes_g = g(x)
        ineq_g.append(gx)
        ineq_dg.append(dg)
        ineq_hes_g.append(hes_g)

    aug_f_x = t*f_x + phi(ineq_g)
    aug_df_x = t*df_x + d_phi(ineq_g, ineq_dg)
    aug_hessian_f_x = t*hessian_f_x + hessian_phi(ineq_g, ineq_dg, ineq_hes_g)
    return aug_f_x, aug_df_x, aug_hessian_f_x


def const_nt_decent(func, x0, eq_constraints_mat, eq_constraints_rhs, t, e=10 ** (-7), init_step_len=1, max_iter=100, slope_ratio=np.e**(-4), back_track_factor=0.2):
    path = []
    step_size = init_step_len
    x_prev = x0

    f_prev, df_prev, hessian_prev = func(x_prev, t)
    i = 0
    success = False

    while not success and i < max_iter:

        p, w = nt_sol(df_prev, hessian_prev, eq_constraints_mat)
        x_next = x_prev + (step_size * p)
        f_next, df_next, hessian_next = func(x_next, t)
        while not wolf_condition(slope_ratio, step_size, f_next, f_prev, df_prev, p):
            step_size = step_size * back_track_factor
            p, w = nt_sol(df_prev, hessian_prev, eq_constraints_mat)
            x_next = x_prev + (step_size * p)
            f_next, df_next, hessian_next = func(x_next,t)

        step_size = init_step_len
        i += 1
        success = check_conv(p ,hessian_next,e)
        #report(i, x_prev, x_next, f_prev, f_next, success)
        path.append(x_prev)
        f_prev = f_next
        df_prev = df_next
        hessian_prev = hessian_next
        x_prev = x_next

    return x_next, success, path




def term_cond(ineq_constraints, t, e):
    if len(ineq_constraints)/t < e:
        return True
    else:
        return False


def phi(in_g):
    phi1 = 0
    for g_x in in_g:
        phi1 += np.log(-g_x)

    return -phi1


def d_phi(in_g, in_dg):
    dphi = np.zeros(in_dg[0].shape)
    for i in range(len(in_dg)):
        dphi += (in_dg[i]*1.0)/(-in_g[i])

    return dphi


def hessian_phi(in_g, in_dg, in_hes_g):
    hes_phi = np.zeros(in_hes_g[0].shape)
    for i in range(len(in_g)):
        dg = in_dg[i].reshape(in_dg[i].shape[0], 1)
        hes_phi += ((1.0/(in_g[i]**2))*np.matmul(dg, np.transpose(dg)) + (in_hes_g[i]*1.0)/(-in_g[i]))

    return hes_phi


def nt_sol(df, hessian, eq_constraints_mat):

    kkt_mat = np.append(hessian, np.transpose(eq_constraints_mat), axis=1)
    temp = np.append(eq_constraints_mat, np.zeros((eq_constraints_mat.shape[0], eq_constraints_mat.shape[0])), axis=1)
    kkt_mat = np.append(kkt_mat, temp, axis=0)
    rhs = np.append(-df, np.zeros(eq_constraints_mat.shape[0]))
    solution = np.linalg.solve(kkt_mat, rhs)
    p = solution[:df.shape[0]]
    w = solution[df.shape[0]:]
    return p, w


def check_conv(p, hessian, e):
    lamda = 0.5*np.dot(p, np.dot(hessian, p))
    if lamda < e:
        return True
    else:
        return False


def newton_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor, t):
    path = []
    step_size = init_step_len
    x_prev = x0
    f_prev, df_prev, hessian_prev = f(x0, t, hessian=True)
    i = 0
    success = False
    while not success and i < max_iter:

        p = np.linalg.solve(hessian_prev, -df_prev)
        x_next = x_prev + (step_size * p)
        f_next, df_next, hessian_next = f(x_next,t, hessian=True)
        while not wolf_condition(slope_ratio, step_size, f_next, f_prev, df_prev, p):
            step_size = step_size*back_track_factor
            p = np.linalg.solve(hessian_prev, -df_prev)
            x_next = x_prev + (step_size * p)
            f_next, df_next, hessian_next = f(x_next,t, hessian=True)

        step_size = init_step_len
        i += 1
        success = check_converge(x_next, x_prev, f_next, f_prev, obj_tol, param_tol)
        #report(i, x_prev, x_next, f_prev, f_next, success)
        path.append(x_prev)
        f_prev = f_next
        df_prev = df_next
        hessian_prev = hessian_next
        x_prev = x_next

    return x_next, success, path