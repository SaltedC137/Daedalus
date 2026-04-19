cmake_minimum_required(VERSION 3.14)


# Enable compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


# Global C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# output file

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# vcpkg toolchain for windows
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_TOOLCHAIN "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        if(EXISTS "${VCPKG_TOOLCHAIN}")
            set(CMAKE_TOOLCHAIN_FILE "${VCPKG_TOOLCHAIN}"
                CACHE STRING "Vcpkg toolchain file")
            message(STATUS "Found vcpkg toolchain: ${VCPKG_TOOLCHAIN}")
        else()
            message(STATUS "VCPKG_ROOT is set but vcpkg toolchain not found at ${VCPKG_TOOLCHAIN}; vcpkg will not be used.")
        endif()
    endif()
endif()


project(DAEDALUS VERSION 0.1.0 LANGUAGES CXX)

find_package(raylib CONFIG REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE raylib)


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

