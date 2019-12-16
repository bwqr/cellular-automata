#include <mpi.h>
#include <iostream>
#include <cmath>
#include <zconf.h>
#include "worker.h"

worker::worker(int rank, int np, int dim, int turn) {

    this->rank = rank;
    this->np = np;
    this->dim = dim;
    this->turn = turn;
    this->worker_dim = (int) std::sqrt(np);
    this->row = (rank - 1) / worker_dim;
    this->column = rank - worker_dim * row - 1;

    cells = new bool *[dim];
    next_cells = new bool *[dim];
    ucells = new bool[dim];
    dcells = new bool[dim];
    rcells = new bool[dim + 2];
    lcells = new bool[dim + 2];

    for (int i = 0; i < dim; ++i) {
        cells[i] = new bool[dim];
        next_cells[i] = new bool[dim];

        ompi_status_public_t status;

        MPI_Recv((void *) cells[i], dim, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, &status);
    }
}

void worker::print() {

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            std::cout << cells[i][j] << this->rank << " ";
        }

        std::cout << std::endl;
    }
}

void worker::run() {
    int up, down, right, left;
    up = find_rank(UP);
    down = find_rank(DOWN),
            right = find_rank(RIGHT);
    left = find_rank(LEFT);

    bool clone_lcells[dim + 2], clone_rcells[dim + 2];

    for (int i = 0; i < turn; ++i) {

        if (is_horizontal_odd()) {
            MPI_Recv(dcells, dim, MPI_CXX_BOOL, down, 0, MPI_COMM_WORLD, nullptr);
            MPI_Send(cells[dim - 1], dim, MPI_CXX_BOOL, down, 0, MPI_COMM_WORLD);
            MPI_Recv(ucells, dim, MPI_CXX_BOOL, up, 0, MPI_COMM_WORLD, nullptr);
            MPI_Send(cells[0], dim, MPI_CXX_BOOL, up, 0, MPI_COMM_WORLD);
        } else {
            MPI_Send(cells[0], dim, MPI_CXX_BOOL, up, 0, MPI_COMM_WORLD);
            MPI_Recv(ucells, dim, MPI_CXX_BOOL, up, 0, MPI_COMM_WORLD, nullptr);
            MPI_Send(cells[dim - 1], dim, MPI_CXX_BOOL, down, 0, MPI_COMM_WORLD);
            MPI_Recv(dcells, dim, MPI_CXX_BOOL, down, 0, MPI_COMM_WORLD, nullptr);
        }

        clone_column(clone_lcells, 0);
        clone_column(clone_rcells, dim - 1);

        if (is_vertical_odd()) {
            MPI_Recv(rcells, dim + 2, MPI_CXX_BOOL, right, 0, MPI_COMM_WORLD, nullptr);
            MPI_Send(clone_rcells, dim + 2, MPI_CXX_BOOL, right, 0, MPI_COMM_WORLD);
            MPI_Recv(lcells, dim + 2, MPI_CXX_BOOL, left, 0, MPI_COMM_WORLD, nullptr);
            MPI_Send(clone_lcells, dim + 2, MPI_CXX_BOOL, left, 0, MPI_COMM_WORLD);
        } else {
            MPI_Send(clone_lcells, dim + 2, MPI_CXX_BOOL, left, 0, MPI_COMM_WORLD);
            MPI_Recv(lcells, dim + 2, MPI_CXX_BOOL, left, 0, MPI_COMM_WORLD, nullptr);
            MPI_Send(clone_rcells, dim + 2, MPI_CXX_BOOL, right, 0, MPI_COMM_WORLD);
            MPI_Recv(rcells, dim + 2, MPI_CXX_BOOL, right, 0, MPI_COMM_WORLD, nullptr);
        }

        simulate();
    }

    send_manager();
}

void worker::send_manager() {
    for (int i = 0; i < dim; ++i) {
        MPI_Send((void *) cells[i], dim, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD);
    }
}

int worker::find_rank(direction dir) {
    int r = row;
    int c = column;

    switch (dir) {
        case UP:
            r = row == 0 ? worker_dim - 1 : row - 1;
            break;
        case DOWN:
            r = row == worker_dim - 1 ? 0 : row + 1;
            break;
        case RIGHT:
            c = column == worker_dim - 1 ? 0 : column + 1;
            break;
        case LEFT:
            c = column == 0 ? worker_dim - 1 : column - 1;
            break;
    }

    return r * worker_dim + c + 1;
}

bool worker::is_horizontal_odd() {
    return (row + 1) % 2 == 1;
}

bool worker::is_vertical_odd() {
    return (column + 1) % 2 == 1;
}

worker::~worker() {
    for (int i = 0; i < dim; ++i) {
        delete[] cells[i];
        delete[] next_cells[i];
    }

    delete[] cells;
    delete[] next_cells;
    delete[] ucells;
    delete[] dcells;
    delete[] lcells;
    delete[] rcells;
}

void worker::clone_column(bool *col, int index) {
    col[0] = ucells[index];

    for (int i = 0; i < dim; ++i) {
        col[i + 1] = cells[i][index];
    }

    col[dim + 1] = dcells[index];
}

void worker::simulate() {

    for (int i = 1; i < dim - 1; ++i) {
        for (int j = 1; j < dim - 1; ++j) {
            next_cells[i][j] = alive_or_dead_central(i, j);
        }

        next_cells[i][0] = alive_or_dead_edge(i, 0);
        next_cells[i][dim - 1] = alive_or_dead_edge(i, dim - 1);
        next_cells[0][i] = alive_or_dead_edge(0, i);
        next_cells[dim - 1][i] = alive_or_dead_edge(dim - 1, i);
    }

    next_cells[0][0] = alive_or_dead_corner(0, 0);
    next_cells[0][dim - 1] = alive_or_dead_corner(0, dim - 1);
    next_cells[dim - 1][0] = alive_or_dead_corner(dim - 1, 0);
    next_cells[dim - 1][dim - 1] = alive_or_dead_corner(dim - 1, dim - 1);

    std::swap(cells, next_cells);
}

bool worker::alive_or_dead_central(int r, int c) {
    int alive_neighbour = (cells[r - 1][c - 1] + cells[r - 1][c] + cells[r - 1][c + 1] + cells[r][c - 1] +
                           cells[r][c + 1] +
                           cells[r + 1][c - 1] + cells[r + 1][c] + cells[r + 1][c + 1]);
    
    return (cells[r][c] && (alive_neighbour == 2 || alive_neighbour == 3)) || (!cells[r][c] && alive_neighbour == 3);
}

bool worker::alive_or_dead_corner(int r, int c) {
    int alive_neighbour = 0;

    if (r == 0) {
        alive_neighbour =
                c == 0 ? lcells[0] + lcells[1] + lcells[2] + ucells[0] + ucells[1] + cells[1][0] + cells[0][1] +
                         cells[1][1]
                       :
                rcells[0] + rcells[1] + rcells[2] + ucells[dim - 1] + ucells[dim - 2] + cells[0][dim - 2] +
                cells[1][dim - 2] + cells[1][dim - 1];
    } else {
        alive_neighbour =
                c == 0 ? lcells[dim + 1] + lcells[dim] + lcells[dim - 1] + dcells[0] + dcells[1] + cells[dim - 1][1] +
                         cells[dim - 2][1] + cells[dim - 2][0]
                       :
                rcells[dim + 1] + rcells[dim] + rcells[dim - 1] + dcells[dim - 1] + dcells[dim - 2] +
                cells[dim - 1][dim - 2] + cells[dim - 2][dim - 2] + cells[dim - 2][dim - 1];
    }

    return (cells[r][c] && (alive_neighbour == 2 || alive_neighbour == 3)) || (!cells[r][c] && alive_neighbour == 3);
}

bool worker::alive_or_dead_edge(int r, int c) {
    int alive_neighbour = 0;

    if (r == 0) {
        alive_neighbour = ucells[c] + ucells[c - 1] + ucells[c + 1] + cells[1][c] + cells[1][c - 1] + cells[1][c + 1] +
                          cells[0][c - 1] + cells[0][c + 1];
    } else if (c == 0) {
        alive_neighbour =
                lcells[r] + lcells[r + 1] + lcells[r + 2] + cells[r - 1][0] + cells[r + 1][0] + cells[r - 1][1] +
                cells[r][1] + cells[r + 1][1];
    } else if (r == dim - 1) {
        alive_neighbour =
                dcells[c] + dcells[c - 1] + dcells[c + 1] + cells[r][c - 1] + cells[r][c + 1] + cells[r - 1][c - 1] +
                cells[r - 1][c] + cells[r - 1][c + 1];
    } else {
        alive_neighbour = rcells[r] + rcells[r + 1] + rcells[r + 2] + cells[r - 1][c - 1] + cells[r][c - 1] +
                          cells[r + 1][c - 1] + cells[r - 1][c] + cells[r + 1][c];
    }

    return (cells[r][c] && (alive_neighbour == 2 || alive_neighbour == 3)) || (!cells[r][c] && alive_neighbour == 3);
}
