from MatrixConverter import *
from LU import LU_CPU, LU_GPU


# Example usage
example_matrix = [
    [10, 0, 0, 0],
    [0, 20, 0, 0],
    [0, 0, 30, 0],
    [0, 0, 0, 40]
]

# Convert to CSR and back
csr_values, csr_row_indices, csr_column_pointers = matrix_to_csr(example_matrix)
restored_matrix_from_csr = csr_to_matrix(csr_values, csr_row_indices, csr_column_pointers)

# Convert to CSC and back
csc_values, csc_column_indices, csc_row_pointers = matrix_to_csc(example_matrix)
restored_matrix_from_csc = csc_to_matrix(csc_values, csc_column_indices, csc_row_pointers)

print("Original Matrix:", example_matrix)
print(f"CSR \nV: {csr_values} \nCOL_INDEX: {csr_row_indices} \nROW_INDEX: {csr_column_pointers}")
print("Restored from CSR:\n", restored_matrix_from_csr)
print(f"CSC \nV: {csc_values} \nCOL_INDEX: {csc_column_indices} \nROW_INDEX: {csc_row_pointers}")
print("Restored from CSC:\n", restored_matrix_from_csc)

print(LU_CPU(example_matrix))
print(LU_GPU(example_matrix))