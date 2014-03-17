from argparse import ArgumentParser
from time import time
import pdb

import numpy as np
import pyopencl as cl


def save_matrix(M, file_name='generated_output.txt'):
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
parser.add_argument('-c', '--check', default=False,
        help=('Validate output against serial numpy implementation.'))
parser.add_argument('-r', '--rtol', default=1e-03,
        help=('Relative tolerance for validation check.'))
parser.add_argument('-a', '--atol', default=1e-05,
        help=('Absolute tolerance for validation check.'))


args = parser.parse_args()
rtol = float(args.rtol)
atol = float(args.atol)

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

N = np.int64(m_A.shape[0])
M = np.int64(m_B.shape[1])
K = np.int64(m_A.shape[1])
assert K == m_B.shape[0], "Inner matrix dimensions do not agree."
m_C = np.zeros(shape=(N, M), dtype=np.float32)

# ----- setup -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

with open('mm.cl', 'r') as f:
    source = "".join(f.readlines())
program = cl.Program(ctx, source).build()

global_size = m_C.shape
#local_size = (16, 4)
local_size = None

# ----- pushing -----
m_A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_A)
m_B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_B)
result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, m_C.nbytes)

# ----- benchmark kernel -----
gpu_start = time()

cycles = 20
for i in range(cycles):
    bench = program.mm(queue, global_size, local_size,
            N, M, K, m_A_buf, m_B_buf, result_buf)
    bench.wait()

gpu_time = (time() - gpu_start) / cycles

# ----- copy back to host -----
cl.enqueue_read_buffer(queue, result_buf, m_C).wait()

# ---- performance data -----
print "Time per multiplication (s): ", gpu_time

# ---- validation -----
if args.check is True:
    serial_C = np.dot(m_A, m_B)

    print "Serial numpy implementation:"
    print serial_C, "\n"
    print "PyOpenCL implementation:"
    print m_C, "\n"
    print "Outputs equal to within rel={0} and abs={1}: {2}".format(
            rtol, atol, np.allclose(serial_C, m_C, rtol, atol))

