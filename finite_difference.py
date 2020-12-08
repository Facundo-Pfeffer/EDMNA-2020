import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from model.models import Vertex


class FiniteDifferenceProblem:
    def __init__(self, x_partitions, y_partitions, x_domain, y_domain,
                 left_x_condition, right_x_condition, bottom_y_condition, top_y_condition,
                 differential_coefficients):

        self.h = self.get_partition(domain=x_domain, partition=x_partitions)
        self.k = self.get_partition(domain=y_domain, partition=y_partitions)

        self.x_domain = x_domain
        self.y_domain = x_domain

        self.x_part = x_partitions
        self.y_part = y_partitions

        self.left_x_condition = left_x_condition
        self.right_x_condition = right_x_condition
        self.bottom_y_condition = bottom_y_condition
        self.top_y_condition = top_y_condition

        self.first_condition = differential_coefficients['G']

        self.coefficient_matrix = self.get_coefficient_matrix(coef_dict=differential_coefficients)
        self.vertex_values = self.get_vertex_values()
        self.unknown_matrix = self.get_unknown_matrix()

        self.solution = self.make_solution()

        self.point_cloud = self.get_point_cloud()

    @staticmethod
    def get_partition(domain, partition):
        return (domain[1] - domain[0]) / partition

    def get_coefficient_matrix(self, coef_dict):
        return [[0, self.number_4(coef_dict), 0],
                [self.number_3(coef_dict), self.number_2(coef_dict), self.number_1(coef_dict)],
                [0, self.number_5(coef_dict), 0]]

    def number_5(self, coef_dict):
        return coef_dict['C'] / (self.k ** 2) - coef_dict['E'] / (2 * self.k)

    def number_4(self, coef_dict):
        return coef_dict['C'] / (self.k ** 2) + coef_dict['E'] / (2 * self.k)

    def number_3(self, coef_dict):
        return (coef_dict['A'] / self.h ** 2) - (coef_dict['D'] / (2 * self.h))

    def number_2(self, coef_dict):
        return (-2 * coef_dict['A'] / self.h ** 2) - (2 * coef_dict['C'] / (self.k ** 2)) + coef_dict['F']

    def number_1(self, coef_dict):
        return coef_dict['A'] / (self.h ** 2) + (coef_dict['D'] / (2 * self.h))

    def general_coef(self):
        """Obtiene la Matriz de coeficientes"""
        h = self.h
        k = self.k
        return [[0, 1 / k ** 2, 0],
                [1 / (h ** 2) + 1 / (2 * h), -2 * (1 / h ** 2 + 1 / k ** 2), 1 / (h ** 2) - 1 / (2 * h)],
                [0, 1 / k ** 2, 0]]

    def get_unknowns_position(self):
        return [(i, j) for i in range(self.y_part) for j in range(self.x_part) if
                i != 0 and i != self.y_part and j != 0 and j != self.x_part]

    def get_vertex_values(self):
        """Obtiene los valores de los vertices y discrimina aquellos que son incognita de aquellos que no lo son"""
        unknowns_positions = self.get_unknowns_position()
        unknown_identifier = 0
        vertex_values = self.initialize_vertex_values()
        for i in range(self.y_part + 1):
            for j in range(self.x_part + 1):
                vertex = Vertex()
                if (i, j) in unknowns_positions:
                    vertex.IsUnknown = True
                    vertex.Position = (i, j)
                    vertex.Value = unknown_identifier
                    unknown_identifier = unknown_identifier + 1
                else:
                    vertex.IsUnknown = False
                    vertex.Position = (i, j)
                    vertex.Value = self.obtain_boundary_values((i, j))
                vertex_values[i][j] = vertex
        return vertex_values

    def initialize_vertex_values(self):
        x_part = self.x_part
        y_part = self.y_part
        """Inicializa la matriz de arriba con todos 0 en las posiciones"""
        result = []
        for i in range(y_part + 1):
            result.append([])
            for j in range(x_part + 1):
                result[i].append(0)
        return result

    def obtain_boundary_values(self, position):
        """Obtiene el valor en los extremos"""
        i = position[0]  # Fila
        j = position[1]  # Columna
        value = None
        if i == 0:  # Extremo superior
            x_position = self.get_position(j)
            value = self.top_y_condition(x_position)
        elif i == self.y_part:  # Extremo inderior
            x_position = self.get_position(j)
            value = self.bottom_y_condition(x_position)
        elif j == 0:  # Extremo izquierdo
            y_position = self.get_position(i, is_x=False)
            value = self.left_x_condition(y_position)
        elif j == self.x_part:  # Extremo derecho:
            y_position = self.get_position(i, is_x=False)
            value = self.right_x_condition(y_position)
        return value

    def get_position(self, index, is_x=True):
        """Devuelve la posición cartesiana de nuestro punto basasdo en nuestros subindices"""
        return self.x_domain[0] + index * self.h if is_x else self.y_domain[1] - index * self.k

    def get_unknown_matrix(self):
        """Devuelve la Matriz de coeficientes y la de terminos independientes de nuestro sistema de ecuaciones"""
        A_matrix = []
        B_matrix = []
        unknown_positions = self.get_unknowns_position()
        for position in unknown_positions:
            row = self.get_unknown_row(position)
            A_matrix.append(row[0])
            B_matrix.append(row[1])
        return (A_matrix, B_matrix)

    def get_unknown_row(self, position):
        """Devuelve los coeficientes y el termino independiente de una ecuación"""
        coefficient_matrix = self.coefficient_matrix
        i = position[0]
        j = position[1]
        row_value = self.initiliaze_row()
        indepent_term = self.first_condition(self.get_position(j), self.get_position(i, is_x=False))
        summ = indepent_term
        for x_mov in (-1, 0, 1):
            for y_mov in (-1, 0, 1):
                vertex = self.vertex_values[i + y_mov][j + x_mov]
                if vertex.IsUnknown:  # Incognita
                    row_value[vertex.Value] = coefficient_matrix[1 + y_mov][1 + x_mov]
                else:
                    summ = summ - coefficient_matrix[1 + y_mov][1 + x_mov] * (vertex.Value)
        return (row_value, summ)

    def initiliaze_row(self):
        """Inicializa nuestra fila, dejando 0 como valor por defecto"""
        return [0 for x in range(len(self.get_unknowns_position()))]

    def make_solution(self):
        """Resuelve el sistema dada ambas matrices"""
        matrix = self.get_unknown_matrix()
        return np.linalg.solve(matrix[0], matrix[1])

    def get_point_cloud(self):
        """Devuelve la nube de puntos con sus valores"""
        point_cloud = self.initialize_vertex_values()
        unknowns_positions = self.get_unknowns_position()
        for i in range(self.y_part + 1):
            for j in range(self.x_part + 1):
                if (i, j) in unknowns_positions:
                    related_vertex = self.vertex_values[i][j]
                    point_cloud[i][j] = self.solution[related_vertex.Value]
                else:
                    point_cloud[i][j] = self.vertex_values[i][j].Value
        return point_cloud

    def plot(self):
        X = []
        Y = []
        Z = []
        for i in range(self.y_part + 1):
            for j in range(self.x_part + 1):
                x_position = self.get_position(j)
                y_position = self.get_position(i, is_x=False)
                X.append(x_position)
                Y.append(y_position)
                Z.append(self.point_cloud[i][j])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='r', marker='o')

        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.set_zlabel('Eje Z')

        plt.show()


problem = FiniteDifferenceProblem(x_domain=(0, 1), y_domain=(0, 1),
                                  x_partitions=15, y_partitions=15,
                                  left_x_condition=lambda y: 0,
                                  right_x_condition=lambda y: 1,
                                  bottom_y_condition=lambda x: x,
                                  top_y_condition=lambda x: x ** 2,
                                  differential_coefficients={'A': 1, #A uxx + B uxy + C uyy + D ux + E uy + F u = G
                                                             'B': 0,
                                                             'C': 1,
                                                             'D': -1,
                                                             'E': 0,
                                                             'F': 0,
                                                             'G': lambda x, y: -1})

# matrix = problem.get_vertex_values()
# print(problem.vertex_values[1][1])
# for i in range(4):
#     ist = [x.Value for x in matrix[i]]
#     print(ist)

matrix_unknown = np.array(problem.get_unknown_matrix()[0])
matrix_independent = np.array(problem.get_unknown_matrix()[1])
# print(f"{matrix_unknown}")
cloud_point = np.array(problem.point_cloud)
# print(cloud_point)

problem.plot()
