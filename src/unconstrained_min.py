import numpy as np
from src.utils import report


def check_converge(x_next, x_prev, f_next, f_prev, obj_tol, param_tol):
    if (np.linalg.norm(x_next-x_prev)) < param_tol or np.linalg.norm(f_next-f_prev) < obj_tol:
        return True
    else:
        return False


def gradient_descent(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor):
    path = []
    step_size = init_step_len
    x_prev = x0
    f_prev, df_prev = f(x0)
    i = 0
    success = False
    while not success and i < max_iter:
        x_next = x_prev - (step_size * df_prev)
        f_next, df_next = f(x_next)
        while not wolf_condition(slope_ratio, step_size, f_next, f_prev, df_prev, -df_prev):
            step_size = step_size*back_track_factor
            x_next = x_prev - (step_size * df_prev)
            f_next, df_next = f(x_next)

        step_size = init_step_len
        i += 1
        success = check_converge(x_next, x_prev, f_next, f_prev, obj_tol, param_tol)
        report(i, x_prev, x_next, f_prev, f_next, success)
        path.append(x_prev)
        f_prev = f_next
        df_prev = df_next
        x_prev = x_next

    return x_next, success, path


def bfgs_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor):
    path = []
    x_prev = x0
    step_size = init_step_len
    f_prev, df_prev = f(x0)
    B_prev = np.identity(df_prev.shape[0])
    i = 0
    success = False
    while not success and i < max_iter:
        p = np.linalg.solve(B_prev, -df_prev)
        x_next = x_prev + (step_size * p)
        f_next, df_next = f(x_next)
        B_next = bfgs(B_prev, df_prev, df_next, x_prev, x_next)
        while not wolf_condition(slope_ratio, step_size, f_next, f_prev, df_prev, p):
            step_size = step_size*back_track_factor
            p = np.linalg.solve(B_prev, -df_prev)
            x_next = x_prev + (step_size * p)
            f_next, df_next = f(x_next)
            B_next = bfgs(B_prev, df_prev, df_next, x_prev, x_next)

        step_size = init_step_len
        i += 1
        success = check_converge(x_next, x_prev, f_next, f_prev, obj_tol, param_tol)
        report(i, x_prev, x_next, f_prev, f_next, success)
        path.append(x_prev)
        f_prev = f_next
        df_prev = df_next
        B_prev = B_next
        x_prev = x_next

    return x_next, success, path


def newton_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor):
    path = []
    step_size = init_step_len
    x_prev = x0
    f_prev, df_prev, hessian_prev = f(x0, hessian=True)
    i = 0
    success = False
    while not success and i < max_iter:

        p = np.linalg.solve(hessian_prev, -df_prev)
        x_next = x_prev + (step_size * p)
        f_next, df_next, hessian_next = f(x_next, hessian=True)
        while not wolf_condition(slope_ratio, step_size, f_next, f_prev, df_prev, p):
            step_size = step_size*back_track_factor
            p = np.linalg.solve(hessian_prev, -df_prev)
            x_next = x_prev + (step_size * p)
            f_next, df_next, hessian_next = f(x_next, hessian=True)

        step_size = init_step_len
        i += 1
        success = check_converge(x_next, x_prev, f_next, f_prev, obj_tol, param_tol)
        report(i, x_prev, x_next, f_prev, f_next, success)
        path.append(x_prev)
        f_prev = f_next
        df_prev = df_next
        hessian_prev = hessian_next
        x_prev = x_next

    return x_next, success, path


def line_search(f, x0, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len=1.0, slope_ratio=np.e**(-4), back_track_factor=0.2):
    if dir_selection_method == 'gd':
        return gradient_descent(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor)
    elif dir_selection_method == 'nt':
        return newton_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor)

    elif dir_selection_method == 'bfgs':
        return bfgs_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor)
    return 0


def bfgs(B_prev, df_prev, df_next, x_prev, x_next):
    sk = x_next - x_prev
    sk = sk.reshape(sk.shape[0], 1)
    if sk.all() == 0:
        return B_prev
    yk = df_next - df_prev
    yk = yk.reshape(yk.shape[0], 1)
    BkskskBK = np.matmul(np.matmul(np.matmul(B_prev, sk), np.transpose(sk)), B_prev)
    skBksk = np.matmul(np.matmul(np.transpose(sk), B_prev), sk)
    B_next = B_prev - (BkskskBK/skBksk) + np.matmul(yk, np.transpose(yk))/np.matmul(np.transpose(yk), sk)
    return B_next


def wolf_condition(slope_ratio, step_size, f_next, f_prev, df_prev, pk):

    if f_next <= f_prev + slope_ratio*step_size*np.matmul(df_prev, pk):
        return True
    else:
        return False
