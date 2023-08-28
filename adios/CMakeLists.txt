cmake_minimum_required(VERSION 3.3)
project(madios)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB SOURCE_FILES *.h *.cpp)
file(GLOB_RECURSE LIB_SOURCE_FILES utils/*.h utils/*.cpp maths/*.h maths/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${LIB_SOURCE_FILES})

# Add executable and link target
add_executable(madios ${SOURCE_FILES})