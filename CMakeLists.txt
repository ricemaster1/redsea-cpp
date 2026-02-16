cmake_minimum_required(VERSION 3.16)
project(RedSea LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ---- GLFW (windowing & input) ----
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
FetchContent_MakeAvailable(glfw)

# ---- GLM (math library) ----
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# ---- OpenGL ----
find_package(OpenGL REQUIRED)

# ---- GLEW (GL loader — Linux/Windows only) ----
if(NOT APPLE)
    find_package(GLEW REQUIRED)
endif()

# ---- Executable ----
add_executable(redsea src/main.cpp)
target_include_directories(redsea PRIVATE src)
target_link_libraries(redsea PRIVATE glfw glm::glm OpenGL::GL)

if(APPLE)
    # Silence Apple's OpenGL deprecation warnings
    target_compile_definitions(redsea PRIVATE GL_SILENCE_DEPRECATION)
else()
    target_link_libraries(redsea PRIVATE GLEW::GLEW)
    target_compile_definitions(redsea PRIVATE USE_GLEW)
endif()

# Release optimisations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(redsea PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -ffast-math>)
endif()
