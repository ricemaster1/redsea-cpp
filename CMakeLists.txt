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

# ---- Executable ----
add_executable(redsea src/main.cpp)
target_include_directories(redsea PRIVATE src)
target_link_libraries(redsea PRIVATE glfw glm::glm OpenGL::GL)

# Silence Apple's OpenGL deprecation warnings
if(APPLE)
    target_compile_definitions(redsea PRIVATE GL_SILENCE_DEPRECATION)
endif()
