cmake_minimum_required(VERSION 3.14)



# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(DAEDALUS VERSION 0.1.0 LANGUAGES CXX)

# Set output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# Compiler optimizations
if(MSVC)
    # Windows / MSVC
    add_compile_options(/O2 /Oi /Ot /GL)
else()
    # Linux/macOS (GCC/Clang)
    # -O3: Maximum optimization
    # -march=native: Architecture specific optimization
    # -ffast-math: Aggressive floating point optimization
    add_compile_options(-march=native)
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


# Set different compiler options based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)

elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME} PRIVATE -O3 -ffast-math)

elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  target_compile_options(${PROJECT_NAME} PRIVATE -O2 -g)
# Asan options
elseif(CMAKE_BUILD_TYPE STREQUAL "Aans")
  target_compile_options(${PROJECT_NAME} PRIVATE -ggdb -O0 -Wall -Wextra -mavx2
      -fsanitize=address
      -fno-omit-frame-pointer
      -fno-optimize-sibling-calls
      -fsanitize-address-use-after-scope
      -fsanitize-recover=address)
endif()


#Asan options
add_link_options(-fsanitize=address)

# copy assets
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/out/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_SOURCE_DIR}/out/assets
    COMMENT "Copying assets folder to out directory"
)

add_custom_target(copy_icon ALL
    DEPENDS ${CMAKE_SOURCE_DIR}/out/assets
)
