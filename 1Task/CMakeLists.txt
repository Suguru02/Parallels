cmake_minimum_required(VERSION 3.21) 

project(first_task) #дает имя проекту


option(USE_DOUBLE "Use double (otherwise use float)" ON)

# В зависимости от опции определяем тип
if(USE_DOUBLE)
    message(STATUS "Using double")
    add_definitions(-DUSE_DOUBLE=1)
    set(TYPE "double")
else()
    message(STATUS "Using float")
    add_definitions(-DUSE_DOUBLE=0)
    set(TYPE "float")
endif()


#дает понять что будет собираться исполняеый файл(exe)
#первый аргумент - это название собираемоего проекта,
#остальные исходные файлы *.h *.cpp
#все аргументы записываются через пробел
add_executable(first_task first.cpp) 

