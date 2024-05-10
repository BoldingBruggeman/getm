cmake_minimum_required(VERSION 3.15)

# Create main project.
project(airsea VERSION 0.1.1 LANGUAGES Fortran C)

# Specify position-independent code since we will build a shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use solution folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_library(cairsea
   src/airsea_fluxes.F90
   src/airsea_variables.F90
   src/humidity.F90
   src/longwave_radiation.F90
   src/fairall.F90
   src/t10m.F90
   src/kara.F90
   src/kondo.F90
   src/solar_zenith_angle.F90
   src/albedo_water.F90
   src/shortwave_radiation.F90
)
target_compile_definitions(cairsea PRIVATE GUSTINESS)

if (MSVC)
  target_compile_options(cairsea PRIVATE /libs:static)
endif()

# Find target Python interpreter.
find_package(Python3 COMPONENTS Interpreter Development NumPy)

add_custom_command(OUTPUT _pyairsea.c
  COMMENT "Creating ${CMAKE_CURRENT_BINARY_DIR}/_pyairsea.c from ${CMAKE_CURRENT_SOURCE_DIR}/_pyairsea.pyx"
  COMMAND Python3::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/_pyairsea.pyx" -o "${CMAKE_CURRENT_BINARY_DIR}/_pyairsea.c"
  DEPENDS _pyairsea.pyx
  VERBATIM
)

Python3_add_library(_pyairsea MODULE
  "${CMAKE_CURRENT_BINARY_DIR}/_pyairsea.c")
target_link_libraries(_pyairsea PRIVATE cairsea Python3::NumPy)

install(TARGETS _pyairsea DESTINATION .)
