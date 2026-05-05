cmake_minimum_required(VERSION 3.10)
project(task2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Принудительно используем GCC
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

add_executable(task2 server.cpp)

target_link_libraries(task2 pthread)