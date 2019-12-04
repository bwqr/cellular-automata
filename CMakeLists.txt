cmake_minimum_required(VERSION 3.14)
project(cellularautomata)

set(CMAKE_CXX_STANDARD 14)

add_executable(cellularautomata main.cpp worker.cpp worker.h defs.h direction.h)