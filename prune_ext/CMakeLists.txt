cmake_minimum_required(VERSION 3.15)
project(prune_ext LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# --- Python discovery ---
# Use the interpreter that invoked cmake (via build.sh) or fall back to PATH.
# Avoids CMake picking up Anaconda's python.app GUI wrapper on macOS.
if(NOT PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE NAMES python3 python REQUIRED)
endif()
message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")

# --- pybind11 discovery ---
# pybind11 is a Python package; ask the active interpreter where its CMake
# config lives rather than requiring a system-wide install.
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
        "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pb11_result)
if(NOT _pb11_result EQUAL 0)
    message(FATAL_ERROR
        "pybind11 not found in the active Python environment.\n"
        "Install it with:  pip install pybind11")
endif()
find_package(pybind11 REQUIRED)

# Apple Clang does not bundle OpenMP — wire Homebrew libomp if available.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(LIBOMP_PREFIX)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP)

pybind11_add_module(prune_ext prune_ext.cpp)

# Place the .so in the repo root so `import prune_ext` works without
# modifying PYTHONPATH — consistent with how other modules are imported.
set_target_properties(prune_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(prune_ext PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(prune_ext PRIVATE USE_OPENMP)
    message(STATUS "OpenMP found — parallel search enabled")
else()
    message(WARNING "OpenMP not found — serial C++ search (still faster than Python due to og_outputs precomputation)")
endif()

target_compile_options(prune_ext PRIVATE -O3 -march=native)
