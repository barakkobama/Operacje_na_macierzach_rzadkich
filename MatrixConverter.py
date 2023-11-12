import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def matrix_to_csr(matrix):
    values = []
    row_indices = []
    column_pointers = [0]

    for row in matrix:
        count_non_zero = 0
        for value in row:
            if value != 0:
                values.append(value)
                row_indices.append(row.index(value))
                count_non_zero += 1
        column_pointers.append(column_pointers[-1] + count_non_zero)

    return values, row_indices, column_pointers

def csr_to_matrix(values, row_indices, column_pointers):
    nrows = len(column_pointers) - 1
    ncols = max(row_indices) + 1 if row_indices else 0
    matrix = np.zeros((nrows, ncols))

    for i in range(nrows):
        start, end = column_pointers[i], column_pointers[i+1]
        for ind in range(start, end):
            j = row_indices[ind]
            matrix[i, j] = values[ind]

    return matrix

def matrix_to_csc(matrix):
    values = []
    column_indices = []
    row_pointers = [0]

    ncols = len(matrix[0])

    for j in range(ncols):
        count_non_zero = 0
        for i, row in enumerate(matrix):
            if row[j] != 0:
                values.append(row[j])
                column_indices.append(i)
                count_non_zero += 1
        row_pointers.append(row_pointers[-1] + count_non_zero)

    return values, column_indices, row_pointers

def csc_to_matrix(values, column_indices, row_pointers):
    ncols = len(row_pointers) - 1
    nrows = max(column_indices) + 1 if column_indices else 0
    matrix = np.zeros((nrows, ncols))

    for j in range(ncols):
        start, end = row_pointers[j], row_pointers[j+1]
        for ind in range(start, end):
            i = column_indices[ind]
            matrix[i, j] = values[ind]

    return matrix

def generate_sparse_matrix(dimensions, zero_percentage, format='list'):
    rows, cols = dimensions
    total_elements = rows * cols
    num_zeros = int((zero_percentage / 100) * total_elements)
    num_nonzeros = total_elements - num_zeros

    elements = np.array([0]*num_zeros + [np.random.randint(1, 10) for _ in range(num_nonzeros)])
    np.random.shuffle(elements)

    matrix = elements.reshape(rows, cols)

    if format == 'csr':
        return csr_matrix(matrix)
    elif format == 'csc':
        return csc_matrix(matrix)
    else:
        return matrix.tolist()