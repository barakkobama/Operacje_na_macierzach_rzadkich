import numpy as np

class MatrixConverter:
    def __init__(self, matrix):
        self.matrix = matrix

    def list_to_matrix(self, matrix_list):
        self.matrix = np.array(matrix_list)

    def numpy_to_list(self):
        self.matrix = self.matrix.tolist()

    def print_matrix(self):
        print(self.matrix)