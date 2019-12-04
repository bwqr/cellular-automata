#include <iostream>
#include <mpi.h>
#include <cmath>
#include <fstream>
#include "worker.h"
#include "defs.h"

using namespace std;

int read_input(char *path, bool **&cells, int dim);

int write_output(char *path, bool **&cells, int dim);

int main(int argc, char **argv) {

    if(argc < 4) {
        cerr << "Three arguments are expected" << endl;
        return -1;
    }

    int rank, size, ret_val = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int worker_dim = (int) sqrt(size);
    int block_dim = DIMENSION / worker_dim;
    int turn = stoi(argv[3]);

    if(rank == 0) {
        bool **cells = nullptr;

        if(read_input(argv[1], cells, DIMENSION) != 0) {
            cerr << "error while reading input file" << endl;
            goto error;
        }

        //Send to workers
        for (int i = 1; i < size; ++i) {
            int row = (i - 1) / worker_dim;
            int column = i - worker_dim * row - 1;

            for (int j = 0; j < block_dim; ++j) {
                MPI_Send((void *)&cells[block_dim * row + j][block_dim * column], block_dim, MPI_CXX_BOOL, i, 0, MPI_COMM_WORLD);
            }
        }

        //Copy from workers
        for (int i = 1; i < size; ++i) {
            int row = (i - 1) / worker_dim;
            int column = i - worker_dim * row - 1;

            for (int j = 0; j < block_dim; ++j) {
                ompi_status_public_t status;

                MPI_Recv((void *)&cells[block_dim * row + j][block_dim * column], block_dim, MPI_CXX_BOOL, i, 0, MPI_COMM_WORLD, &status);
            }
        }

        if(write_output(argv[2], cells, DIMENSION) != 0) {
            cerr << "error while writing output" << endl;
            goto error;
        }

        goto success;

        error:
            ret_val = -1;

        success:
        // Delete all the data
        for (int i = 0; i < DIMENSION; ++i) {
            delete[] cells[i];
        }

        delete[] cells;

    } else {
        worker worker(rank, size, block_dim, turn);

        worker.run();
    }

    MPI_Finalize();

    return ret_val;
}

int read_input(char *path, bool **&cells, int dim) {
    ifstream ifstream(path);

    if(!ifstream.is_open()) {
        return -1;
    }

    cells = new bool*[dim];

    for (int i = 0; i < dim; ++i) {
        cells[i] = new bool[dim];

        for (int j = 0; j < dim; ++j) {
            char ch;

            ifstream >> ch;

            cells[i][j] = ch == '1';
        }
    }

    return 0;
}

int write_output(char *path, bool **&cells, int dim) {

    ofstream ofstream(path);

    if(!ofstream.is_open()) {
        return -1;
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            ofstream << cells[i][j] << " ";
        }

        ofstream << endl;
    }

    return 0;
}