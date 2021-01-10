cmake_minimum_required(VERSION 3.17)
project(task_1_transputer_matrix_MPI)

find_package(MPI)
add_definitions(-DOMPI_SKIP_MPICXX)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(main main.cpp)

target_link_libraries(main PRIVATE MPI::MPI_C)

set(CMAKE_CXX_STANDARD 17)
