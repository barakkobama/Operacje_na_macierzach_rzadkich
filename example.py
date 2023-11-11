from MatrixConverter import MatrixConverter
from LU import LU_CPU, LU_GPU


matrix_list = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

converter = MatrixConverter(matrix_list)

# Wyświetlanie macierzy w oryginalnej reprezentacji
print("Original Matrix:")
converter.print_matrix()

# Konwersja listy na macierz NumPy
converter.list_to_matrix(matrix_list)
print("\nNumPy Matrix:")
converter.print_matrix()

# Konwersja macierzy NumPy na listę
converter.numpy_to_list()
print("\nList Matrix:")
converter.print_matrix()

print(LU_CPU(matrix_list))
print(LU_GPU(matrix_list))