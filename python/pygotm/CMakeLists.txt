# Ensure FindPython3 supports NumPy
cmake_minimum_required(VERSION 3.14)

# Create main project.
project(pygotm VERSION 0.1.0 LANGUAGES Fortran C)

# Specify position-independent code since we will build a shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use solution folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (MSVC)
  add_compile_options(/libs:static)
endif()

# Find target Python interpreter.
set(Python_FIND_STRATEGY LOCATION)
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/modules")

add_subdirectory(../../extern/pygsw pygsw EXCLUDE_FROM_ALL)

set(FLEXOUT_USE_NetCDF OFF CACHE BOOL "Enable output in NetCDF format" FORCE)
mark_as_advanced(FLEXOUT_USE_NetCDF)
add_subdirectory(../../extern/gotm/extern/flexout flexout EXCLUDE_FROM_ALL)

add_subdirectory(../../extern/gotm/src/util gotm/util EXCLUDE_FROM_ALL)
add_subdirectory(../../extern/gotm/src/turbulence gotm/turbulence EXCLUDE_FROM_ALL)
target_include_directories(turbulence PRIVATE ../../extern/gotm/include)
target_include_directories(util PRIVATE ../../extern/gotm/include)
if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
  target_compile_options(turbulence PRIVATE -ffree-line-length-none)
  target_compile_options(util PRIVATE -ffree-line-length-none)
endif()

add_library(cgotm
  pygotm_wrapper.F90)
target_link_libraries(cgotm turbulence util yaml)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_pygotm.c
  COMMAND cython _pygotm.pyx
  MAIN_DEPENDENCY _pygotm.pyx
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

Python3_add_library(_pygotm MODULE
  _pygotm.c)
target_link_libraries(_pygotm PRIVATE cgotm Python3::NumPy)

install(TARGETS _pygotm DESTINATION .)
