from Matrix_Converter import *
from Matrix_Generator import *
from LU import LU_CPU, LU_GPU

# Example matrix from genrator
#example_matrix = generate_sparse_matrix((10,10), 80, 'list')

# Example matrix
example_matrix = np.array([
    [10, 0, 0, 0],
    [0, 20, 0, 0],
    [0, 0, 30, 0],
    [0, 0, 0, 40]
], dtype=np.float32)

# Convert to CSR using GPU and print
values_gpu, row_indices_gpu, column_pointers_gpu = matrix_to_csr_gpu(example_matrix)
print("CSR Format (GPU):")
print("Values:", values_gpu)
print("Row Indices:", row_indices_gpu)
print("Column Pointers:", column_pointers_gpu)

# Convert to CSS using GPU and print
values_gpu, row_indices_gpu, column_pointers_gpu = matrix_to_csc_gpu(example_matrix)
print("CSC Format (GPU):")
print("Values:", values_gpu)
print("Row Indices:", row_indices_gpu)
print("Column Pointers:", column_pointers_gpu)

print("Original Matrix:")

print(LU_CPU(example_matrix))
print(LU_GPU(example_matrix))