import numpy as np

class MatrixConverter:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def list_to_matrix(self, matrix_list):
        self.matrix = np.array(matrix_list)

    def numpy_to_list(self):
        return self.matrix.tolist()

    def print_matrix(self):
        print(self.matrix)

    def to_csr(self):
        # Convert to CSR format
        values = []
        row_indices = []
        column_pointers = [0]

        for i, row in enumerate(self.matrix):
            for j, value in enumerate(row):
                if value != 0:
                    values.append(value)
                    row_indices.append(j)
            column_pointers.append(len(values))

        return values, row_indices, column_pointers

    def from_csr(self, values, row_indices, column_pointers):
        # Convert from CSR format
        nrows = len(column_pointers) - 1
        ncols = max(row_indices) + 1 if row_indices else 0
        self.matrix = np.zeros((nrows, ncols))

        for i in range(nrows):
            start, end = column_pointers[i], column_pointers[i+1]
            for ind in range(start, end):
                j = row_indices[ind]
                self.matrix[i, j] = values[ind]

    def to_csc(self):
        # Convert to CSC format
        values = []
        column_indices = []
        row_pointers = [0]
        ncols = self.matrix.shape[1]

        for j in range(ncols):
            for i, row in enumerate(self.matrix):
                if row[j] != 0:
                    values.append(row[j])
                    column_indices.append(i)
            row_pointers.append(len(values))

        return values, column_indices, row_pointers

    def from_csc(self, values, column_indices, row_pointers):
        # Convert from CSC format
        ncols = len(row_pointers) - 1
        nrows = max(column_indices) + 1 if column_indices else 0
        self.matrix = np.zeros((nrows, ncols))

        for j in range(ncols):
            start, end = row_pointers[j], row_pointers[j+1]
            for ind in range(start, end):
                i = column_indices[ind]
                self.matrix[i, j] = values[ind]

    def lu_decomposition(self):
        # Perform LU decomposition with partial pivoting
        n = self.matrix.shape[0]
        L = np.eye(n)
        U = self.matrix.astype(float)
        
        for i in range(n):
            for j in range(i+1, n):
                L[j, i] = U[j, i] / U[i, i]
                U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]
                
        return L, U
