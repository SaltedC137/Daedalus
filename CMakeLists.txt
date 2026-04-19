cmake_minimum_required(VERSION 3.14)

# 1. Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# 2. Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 3. Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(DAEDALUS VERSION 0.1.0 LANGUAGES CXX)

# Set output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# 4. Compiler optimizations
if(MSVC)
    # Windows / MSVC
    add_compile_options(/O2 /Oi /Ot /GL)
else()
    # Linux/macOS (GCC/Clang)
    # -O3: Maximum optimization
    # -march=native: Architecture specific optimization
    # -ffast-math: Aggressive floating point optimization
    add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(raylib CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS 
  src/train.cpp 
  src/mat.cpp 
  src/net.cpp
)
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE raylib Threads::Threads OpenMP::OpenMP_CXX)


# Include dirc
target_include_directories(${PROJECT_NAME} 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3party>
)


# copy assets
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/out/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_SOURCE_DIR}/out/assets
    COMMENT "Copying assets folder to out directory"
)

add_custom_target(copy_icon ALL
    DEPENDS ${CMAKE_SOURCE_DIR}/out/assets
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /O2)
endif()

