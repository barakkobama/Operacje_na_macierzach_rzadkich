from MatrixConverter import MatrixConverter


# Example usage
matrix_list = [
    [4, 0, 0],
    [0, 5, 0],
    [0, 0, 6]
]
converter = MatrixConverter(matrix_list)

# Convert to CSR and print
values, row_indices, column_pointers = converter.to_csr()
print("CSR Format:", values, row_indices, column_pointers)

# Convert back from CSR and print
converter.from_csr(values, row_indices, column_pointers)
converter.print_matrix()

# LU Decomposition
L, U = converter.lu_decomposition()
print("L Matrix:\n", L)
print("U Matrix:\n", U)