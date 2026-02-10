cmake_minimum_required(VERSION 3.21) 

project(first_task) #дает имя проекту


option(USE_DOUBLE "Use double (otherwise use float)" ON)

# В зависимости от опции определяем тип
if(USE_DOUBLE)
    add_definitions(-DUSE_DOUBLE=1)
else()
    add_definitions(-DUSE_DOUBLE=0)
endif()

add_executable(first_task first.cpp) 

