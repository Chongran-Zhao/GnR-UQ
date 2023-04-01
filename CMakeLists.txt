cmake_minimum_required(VERSION 3.0)

project(gnr1d)

set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_CXX_STANDARD 11)

# Look for MPI package
find_package(MPI REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "-g -Wall -O3")

set(SOURCES
  ./Model_wall.hpp 
  ./Model_wall.cpp 
  ./Sys_Tools.hpp 
  ./Sys_Tools.cpp 
  ./Time_solver.hpp 
  ./Time_solver.cpp
)

include_directories(${MPI_INCLUDE_PATH})

add_library(gnr1d_lib ${SOURCES})

add_executable(gnr1d main.cpp)

target_link_libraries(gnr1d gnr1d_lib ${MPI_LIBRARIES})
