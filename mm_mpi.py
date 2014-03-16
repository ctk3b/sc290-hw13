from __future__ import division
from argparse import ArgumentParser
import sys
import pdb

import numpy as np
#from mpi4py import MPI

NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3

def save_matrix(M, file_name='lawl.txt'):
    dimensions = [str(dim) for dim in M.shape]
    header = "\t".join(dimensions)
    np.savetxt(file_name, M, fmt='%.6f', delimiter='\t',
            header=header, comments='')

parser = ArgumentParser(description="Matrix multiplication using MPI.")

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
                m_A = np.ndarray(shape=dimensions, dtype=float)
            elif i < m_A.shape[0] + 1:
                row = [float(entry) for entry in line.split()]
                m_A[i - 1] = row
            # Second matrix...
            elif i == m_A.shape[0] + 2:
                dimensions = [int(dim) for dim in line.split()]
                m_B = np.ndarray(shape=dimensions, dtype=float)
            else:
                row = [float(entry) for entry in line.split()]
                m_B[i - (m_A.shape[0] + 3)] = row

if __name__ == "__main__":
    m_C = np.dot(m_A, m_B)
    save_matrix(m_C)
    sys.exit(0)
    comm = MPI.COMM_WORLD
    mpi_rows = int(np.floor(np.sqrt(comm.size)))
    mpi_cols = comm.size // mpi_rows
    if mpi_rows*mpi_cols > comm.size:
        mpi_cols -= 1
    if mpi_rows*mpi_cols > comm.size:
        mpi_rows -= 1

    ccomm = comm.Create_cart( (mpi_rows, mpi_cols), periods=(True, True), reorder=True)

    my_mpi_row, my_mpi_col = ccomm.Get_coords( ccomm.rank )
    neigh = [0,0,0,0]

    neigh[NORTH], neigh[SOUTH] = ccomm.Shift(0, 1)
    neigh[EAST],  neigh[WEST]  = ccomm.Shift(1, 1)


    # Create matrices
    my_A = np.random.normal(size=(my_N, my_M)).astype(np.float32)
    my_B = np.random.normal(size=(my_N, my_M)).astype(np.float32)
    my_C = np.zeros_like(my_A)

    tile_A = my_A
    tile_B = my_B
    tile_A_ = np.empty_like(my_A)
    tile_B_ = np.empty_like(my_A)
    req = [None, None, None, None]

    t0 = time()
    for r in xrange(mpi_rows):
        req[EAST]  = ccomm.Isend(tile_A , neigh[EAST])
        req[WEST]  = ccomm.Irecv(tile_A_, neigh[WEST])
        req[SOUTH] = ccomm.Isend(tile_B , neigh[SOUTH])
        req[NORTH] = ccomm.Irecv(tile_B_, neigh[NORTH])

        #t0 = time()
        my_C += np.dot(tile_A, tile_B)
        #t1 = time()

        req[0].Waitall(req)
        #t2 = time()
        #print("Time computing %6.2f  %6.2f" % (t1-t0, t2-t1))
    comm.barrier()
    t_total = time()-t0

    np.dot(tile_A, tile_B)

    comm.barrier()






