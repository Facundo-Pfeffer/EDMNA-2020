import pytest
from finite_difference import FiniteDifferenceProblem

def problem_1():
    return FiniteDifferenceProblem(x_domain=(0, 1), y_domain=(0, 1),
                                  x_partitions=3, y_partitions=3,
                                  left_x_condition=lambda y: 0,
                                  right_x_condition=lambda y: 1,
                                  bottom_y_condition=lambda x: x,
                                  top_y_condition=lambda x: x ** 2,
                                  differential_coefficients={'A': 1,
                                                             'B': 0,
                                                             'C': 1,
                                                             'D': -1,
                                                             'E': 0,
                                                             'F': 0,
                                                             'G': lambda x, y: -1})

def test_object_definition():
    problem = problem_1()
    assert problem.get_partition(domain=(0, 1), partition=3) == 1/3


def test_coefficient_matrix():
    problem = problem_1()
    h = problem.get_partition(domain=(0, 1), partition=3)
    k = h
    matrix = problem.get_coefficient_matrix()
    print(x for x in matrix)
    assert  matrix == [[0, 9.0, 0], [10.5, -36, 7.5], [0, 9.0, 0]]

def test_number_of_unknowns():
    problem = problem_1()
    unknowns =  problem.get_unknowns_position()
    assert unknowns == [(1,1), (1,2), (2,1), (2,2)]


def test_initialize_vertex_values():
    problem = problem_1()
    assert problem.initialize_vertex_values() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_lambda_functions():
    problem = problem_1()
    assert problem.top_y_condition(2) == 4


def test_initialize_row():
    problem = problem_1()
    assert problem.initiliaze_row() == [0, 0, 0, 0]

def test_general():
    problem = problem_1()
    assert problem.general_coef(coef_dict={'A': 1,
                                      'B': 0,
                                      'C': 1,
                                      'D': -1,
                                      'E': 0,
                                      'F': 0}) == problem.coefficient_matrix

def test_final_result():
    problem = problem_1()
    assert problem.get_unknown_matrix()