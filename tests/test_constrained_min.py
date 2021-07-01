import tests.examples as exmaples
from src.constrained_min import interior_pt
import numpy as np
import src.utils

def test_lp():
    x0 = np.array([0.5, 0.75])
    obj_func = exmaples.obj_lp
    ineq_constraints = [exmaples.lp_ineq_1, exmaples.lp_ineq_2, exmaples.lp_ineq_3, exmaples.lp_ineq_4]
    x_opt, path = interior_pt(obj_func, x0, ineq_constraints)
    print(x_opt)
    print(len(path))

    src.utils.plot_feasibile_lp(exmaples.min_obj_lp, path, title=f"test_quad_min: {obj_func.__name__}")


def test_qp():
    x0 = np.array([0.1, 0.2, 0.7])
    obj_func = exmaples.obj_qp
    ineq_constraints = [exmaples.qp_ineq_1, exmaples.qp_ineq_2, exmaples.qp_ineq_3]
    eq = np.array([1, 1, 1])
    eq = eq.reshape(1, 3)
    x_opt, path = interior_pt(obj_func, x0, ineq_constraints, eq_constraints_mat=eq)
    print(x_opt)
    print(len(path))
    src.utils.plot_feasibile_qp(obj_func, path)
    #src.utils.plotting(obj_func, path, title=f"test_quad_min: {obj_func.__name__}")

test_qp()
