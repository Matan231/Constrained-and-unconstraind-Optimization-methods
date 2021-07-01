import tests.examples as exmaples
from src.constrained_min import interior_pt
import numpy as np
import src.utils
import unittest

class test_constraind_min(unittest.TestCase):


    def test_lp(self):
        x0 = np.array([0.5, 0.75])
        obj_func = exmaples.obj_lp
        ineq_constraints = [exmaples.lp_ineq_1, exmaples.lp_ineq_2, exmaples.lp_ineq_3, exmaples.lp_ineq_4]
        x_opt, path, success= interior_pt(obj_func, x0, ineq_constraints)
        print("test_lp:")
        print(f"optimum x: {x_opt}")
        print(f"optimum f(x): {exmaples.min_obj_lp(x_opt)[0]}")
        src.utils.plot_feasibile_lp(exmaples.min_obj_lp, path, title=f"test_quad_min: {obj_func.__name__}")
        self.assertTrue(success)


    def test_qp(self):
        x0 = np.array([0.1, 0.2, 0.7])
        obj_func = exmaples.obj_qp
        ineq_constraints = [exmaples.qp_ineq_1, exmaples.qp_ineq_2, exmaples.qp_ineq_3]
        eq = np.array([1, 1, 1])
        eq = eq.reshape(1, 3)
        x_opt, path, success = interior_pt(obj_func, x0, ineq_constraints, eq_constraints_mat=eq)
        print("test_qp:")
        print(f"optimum x: {x_opt}")
        print(f"optimum f(x): {obj_func(x_opt)[0]}")
        src.utils.plot_feasibile_qp(obj_func, path)
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()

