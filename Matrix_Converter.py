import numpy as np
import pyopencl as cl

# Initialize OpenCL context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Function: Convert matrix to CSR format using PyOpenCL (GPU)
def matrix_to_csr_gpu(matrix):
    rows, cols = matrix.shape

    # Count non-zero elements per row on CPU
    nnz_per_row = np.sum(matrix != 0, axis=1)
    cum_nnz_per_row = np.zeros(rows + 1, dtype=np.int32)
    cum_nnz_per_row[1:] = np.cumsum(nnz_per_row)
    total_nnz = cum_nnz_per_row[-1]

    # Create an additional buffer to track the current count of non-zero elements per row
    current_row_nnz = np.zeros(rows, dtype=np.int32)

    # Define OpenCL kernel for matrix_to_csr_gpu
    kernel_code = """
    __kernel void matrix_to_csr(__global const float *matrix, __global float *values,
                                __global int *row_indices, __global const int *cum_nnz_per_row,
                                __global int *current_row_nnz, const int cols) {
        int idx = get_global_id(0);
        int row = idx / cols;
        int col = idx % cols;

        if (row < cols && col < cols && matrix[idx] != 0) {
            int nnz_idx = atomic_add(&current_row_nnz[row], 1) + cum_nnz_per_row[row];
            values[nnz_idx] = matrix[idx];
            row_indices[nnz_idx] = col;
        }
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Prepare data and buffers
    matrix_flat = matrix.flatten()
    values = np.zeros(total_nnz, dtype=np.float32)
    row_indices = np.zeros(total_nnz, dtype=np.int32)
    cum_nnz_per_row_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cum_nnz_per_row)
    current_row_nnz_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_row_nnz)
    matrix_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_flat)
    values_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, values.nbytes)
    row_indices_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, row_indices.nbytes)

    # Execute kernel
    global_size = (matrix.size,)
    program.matrix_to_csr(queue, global_size, None, matrix_buf, values_buf, row_indices_buf, cum_nnz_per_row_buf, current_row_nnz_buf, np.int32(cols))

    # Read results
    cl.enqueue_copy(queue, values, values_buf)
    cl.enqueue_copy(queue, row_indices, row_indices_buf)

    return values, row_indices, cum_nnz_per_row

def matrix_to_csc_gpu(matrix):
    rows, cols = matrix.shape

    # Count non-zero elements per column on CPU
    nnz_per_col = np.sum(matrix != 0, axis=0)
    cum_nnz_per_col = np.zeros(cols + 1, dtype=np.int32)
    cum_nnz_per_col[1:] = np.cumsum(nnz_per_col)
    total_nnz = cum_nnz_per_col[-1]

    # Create an additional buffer to track the current count of non-zero elements per column
    current_col_nnz = np.zeros(cols, dtype=np.int32)

    # Define OpenCL kernel for matrix_to_csc_gpu
    kernel_code = """
    __kernel void matrix_to_csc(__global const float *matrix, __global float *values,
                                __global int *column_indices, __global const int *cum_nnz_per_col,
                                __global int *current_col_nnz, const int rows, const int cols) {
        int idx = get_global_id(0);
        int row = idx / cols;
        int col = idx % cols;

        if (row < rows && col < cols && matrix[idx] != 0) {
            int nnz_idx = atomic_add(&current_col_nnz[col], 1) + cum_nnz_per_col[col];
            values[nnz_idx] = matrix[idx];
            column_indices[nnz_idx] = row;
        }
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Prepare data and buffers
    matrix_flat = matrix.flatten()
    values = np.zeros(total_nnz, dtype=np.float32)
    column_indices = np.zeros(total_nnz, dtype=np.int32)
    cum_nnz_per_col_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cum_nnz_per_col)
    current_col_nnz_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_col_nnz)
    matrix_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_flat)
    values_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, values.nbytes)
    column_indices_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, column_indices.nbytes)

    # Execute kernel
    global_size = (matrix.size,)
    program.matrix_to_csc(queue, global_size, None, matrix_buf, values_buf, column_indices_buf, cum_nnz_per_col_buf, current_col_nnz_buf, np.int32(rows), np.int32(cols))

    # Read results
    cl.enqueue_copy(queue, values, values_buf)
    cl.enqueue_copy(queue, column_indices, column_indices_buf)

    return values, column_indices, cum_nnz_per_col