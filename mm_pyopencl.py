from argparse import ArgumentParser
import pdb

import numpy as np
import pyopencl as cl


def save_matrix(M, file_name='lawl.txt'):
    dimensions = [str(dim) for dim in M.shape]
    header = "\t".join(dimensions)
    np.savetxt(file_name, M, fmt='%.6f', delimiter='\t',
            header=header, comments='')

# ----- read input data -----
parser = ArgumentParser(description="Matrix multiplication using OpenCL.")
parser.add_argument('-i', '--infile', default='input.txt',
        help=('Input file containing two matrices, separated by a blank line. '
            'Each matrix is specified as follows: the first line contains two '
            'integers, R and C, separated by TAB, indicating the dimension of '
            'the matrix. It is followed by R lines, each containing C floating'
            ' numbers (TAB separated).'))

args = parser.parse_args()

with open(args.infile, 'r') as f:
    for i, line in enumerate(f):
        if line.strip():  # Toss the blank line.
            # First matrix...
            if i == 0:
                dimensions = [int(dim) for dim in line.split()]
                m_A = np.ndarray(shape=dimensions, dtype=np.float32)
            elif i < m_A.shape[0] + 1:
                row = [float(entry) for entry in line.split()]
                m_A[i - 1] = row
            # Second matrix...
            elif i == m_A.shape[0] + 2:
                dimensions = [int(dim) for dim in line.split()]
                m_B = np.ndarray(shape=dimensions, dtype=np.float32)
            else:
                row = [float(entry) for entry in line.split()]
                m_B[i - (m_A.shape[0] + 3)] = row

N = m_A.shape[0]
M = m_B.shape[1]
K = m_A.shape[1]
assert K == m_B.shape[0], "Inner matrix dimensions do not agree."
print N, M, K

m_B = m_B.T
m_C = np.zeros(shape=(N, M), dtype=np.float32)

# ----- do the magic -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
m_A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_A)
m_B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_B)
result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, m_C.nbytes)

with open('mm.cl', 'r') as f:
    source = "".join(f.readlines())
program = cl.Program(ctx, source).build()

program.mm(queue, m_C.shape, None,
        np.int32(N), np.int32(M), np.int32(K), m_A_buf, m_B_buf, result_buf)
cl.enqueue_read_buffer(queue, result_buf, m_C).wait()

serial_C = np.dot(m_A, m_B.T)

print serial_C
print m_C
