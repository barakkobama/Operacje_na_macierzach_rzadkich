from scipy.sparse import csr_matrix, csc_matrix
import numpy as np

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