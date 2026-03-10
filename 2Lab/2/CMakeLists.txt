cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(task2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_executable(task2 2task.cpp)

target_link_libraries(task2 PRIVATE OpenMP::OpenMP_CXX)