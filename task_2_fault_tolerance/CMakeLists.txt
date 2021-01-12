cmake_minimum_required(VERSION 3.17)
project(Artem)

set(CMAKE_C_COMPILER "/usr/local/Cellar/mpich/3.3.2_1/bin/mpicc")

find_package(MPI)
add_definitions(-DOMPI_SKIP_MPICXX)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(MPI_Version_fdtd_2d MPI_Version/fdtd-2d.c)
add_executable(original_program_fdtd_2d original_program/fdtd-2d.c)
add_executable(MPI_Version_fault_tolerance_fdtd_2d MPI_Version_fault_tolerance/fdtd-2d.c)

