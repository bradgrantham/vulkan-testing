cmake_minimum_required(VERSION 3.12.2)

project(vulkan_test)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

find_package(Vulkan REQUIRED)


find_package(glfw3 3.3 CONFIG) #  REQUIRED)

if(NOT glfw3_FOUND)
    find_package(GLFW3 3.3 CONFIG) # REQUIRED)
    if(NOT GLFW3_FOUND)
        message(
            FATAL_ERROR
            "GLFW3 package was not found under name GLFW3 nor glfw3"
        )
    endif()
endif()

if(NOT Vulkan_FOUND)
    message(
        FATAL_ERROR
            "Vulkan package was not found"
    )
endif()

add_executable(vulkan_test main.cpp vectormath.cpp manipulator.cpp)
target_link_libraries(vulkan_test glfw)
target_link_libraries(vulkan_test ${Vulkan_LIBRARIES})
target_include_directories(vulkan_test PRIVATE ${Vulkan_INCLUDE_DIRS})
# target_include_directories(vulkan_test PRIVATE ${GLFW_INCLUDE_DIR})
set_property(TARGET vulkan_test PROPERTY CXX_STANDARD 20)

if(MSVC)
  target_compile_options(vulkan_test PRIVATE /W4 /WX)
else()
  target_compile_options(vulkan_test PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers)
endif()

