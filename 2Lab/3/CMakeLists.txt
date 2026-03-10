cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(task3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_executable(task3 3task.cpp)

target_link_libraries(task3 PRIVATE OpenMP::OpenMP_CXX)