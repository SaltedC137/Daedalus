cmake_minimum_required(VERSION 3.14)


# Enable compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


# Global C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# output file

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# vcpkg toolchain
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
    message("find vcpkg")
endif()


project(DAEDALUS VERSION 0.1.0 LANGUAGES CXX)

find_package(raylib CONFIG REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE raylib)


# copy assets

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/out/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_SOURCE_DIR}/out/assets
    COMMENT "Copying icon folder to out directory"
)

add_custom_target(copy_icon ALL
    DEPENDS ${CMAKE_SOURCE_DIR}/out/assets
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /O2)
endif()

