cmake_minimum_required(VERSION 3.3)
project(adios)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB SOURCE_FILES *.h *.cpp)
file(GLOB_RECURSE LIB_SOURCE_FILES utils/*.h utils/*.cpp math/*.h math/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${LIB_SOURCE_FILES})

# Add executable and link target
add_executable(adios ${SOURCE_FILES})
