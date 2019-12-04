#ifndef CELLULARAUTOMATA_WORKER_H
#define CELLULARAUTOMATA_WORKER_H


#include "direction.h"

class worker {
public:
    worker(int rank, int np, int dim, int turn);

    ~worker();

    void run();

private:
    bool *ucells;
    bool *dcells;
    bool *rcells;
    bool *lcells;
    bool **next_cells;
    bool **cells;
    int rank;
    int row;
    int column;
    int np;
    int worker_dim;
    int dim;
    int turn;

    void print();

    bool is_horizontal_odd();
    bool is_vertical_odd();

    void clone_column(bool *col, int index);

    void simulate();

    bool alive_or_dead_central(int r, int c);
    bool alive_or_dead_edge(int r, int c);
    bool alive_or_dead_corner(int r, int c);

    int find_rank(direction dir);

    void send_manager();
};


#endif //CELLULARAUTOMATA_WORKER_H
