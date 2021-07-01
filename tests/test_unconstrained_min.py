from src.unconstrained_min import line_search
import tests.examples as exmaples
import src.utils
import numpy as np
import unittest


class test_unconstraind_min(unittest.TestCase):

    Method = 'bfgs'

    def test_quad_min(self, method=Method):
        starting_point = np.array([1, 1])
        step_size = 0.1
        max_iterations = 100
        step_tol = 10**(-8)
        obj_tol = 10**(-12)
        quadratic_functions = [exmaples.f_1, exmaples.f_2, exmaples.f_3]
        for func in quadratic_functions:
            last_point, success, path = line_search(func, starting_point, obj_tol, step_tol, max_iterations, dir_selection_method= method)
            src.utils.plotting(func,path, title = f"test_quad_min: {func.__name__}")
            print(f"{last_point} , {success}")
            self.assertTrue(success)

    def test_rosenbrock_min(self, method=Method):
        starting_point = np.array([2, 2])
        step_size = 1
        max_iterations = 10000
        step_tol = 10 ** (-8)
        obj_tol = 10 ** (-7)
        last_point, success, path = line_search(exmaples.Rosenbrock, starting_point, obj_tol, step_tol, max_iterations, dir_selection_method=method)
        src.utils.plotting(exmaples.Rosenbrock, path, title="test_rosenbrock_min")
        print(f"{last_point} , {success}")
        self.assertTrue(success)


def test_len_min(method='gd'):
    starting_point = np.array([1, 1])
    step_size = 0.1
    max_iterations = 100
    step_tol = 10 ** (-8)
    obj_tol = 10 ** (-12)
    linear_function = exmaples.f_linear
    last_point, success, path = line_search(linear_function, starting_point, obj_tol, step_tol, max_iterations, dir_selection_method=method)
    src.utils.plotting(linear_function, path, plot_range=30 , num_of_counter_lines=15 , title="test_len_min")
    print(f"{last_point} , {success}")

    #test_quad_min(method='nt')
    #test_rosenbrock_min(method='nt')
    #test_len_min()

if __name__ == '__main__':
    unittest.main()
