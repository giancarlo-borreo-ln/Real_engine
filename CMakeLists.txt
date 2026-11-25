cmake_minimum_required(VERSION 3.20)
project(real_engine LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(REAL_ENGINE_AVX2 "Compile AVX2 SIMD paths" OFF)

add_executable(real_engine
    src/re_core.c
    src/re_thread.c
    src/re_mem.c
    src/re_fixed.c
    src/re_vec.c
    src/re_ecs.c
    src/re_scheduler.c
    src/re_bvh.c
    src/re_gjk.c
    src/re_world.c
    src/re_solver.c
    src/re_hash.c
    src/re_bitstream.c
    src/re_proto.c
    src/re_rollback.c
    src/re_render_graph.c
    src/re_renderer.c
    src/re_renderer_frame.c
    src/re_null.c
    src/re_engine.c
    src/main.c
)

target_include_directories(real_engine PRIVATE src)
target_compile_options(real_engine PRIVATE
    -Wall -Wextra -Werror -pedantic -Wconversion -ffp-contract=off
)

# Determinism contract: never contract to FMA, never reassociate.
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(real_engine PRIVATE -ffp-contract=off)
endif()

if(REAL_ENGINE_AVX2)
    target_compile_options(real_engine PRIVATE -mavx2 -DRE_USE_AVX2)
endif()

find_library(MATH_LIB m REQUIRED)
target_link_libraries(real_engine PRIVATE ${MATH_LIB})

find_package(Threads REQUIRED)
target_link_libraries(real_engine PRIVATE Threads::Threads)

target_link_libraries(real_engine PRIVATE ${CMAKE_DL_LIBS})

# Vulkan + X11 backends when the OS headers exist (loader still dlopen'd).
find_path(VULKAN_HEADER vulkan/vulkan.h)
if(VULKAN_HEADER)
    target_compile_definitions(real_engine PRIVATE RE_HAS_VULKAN)
    find_path(X11_HEADER X11/Xlib.h)
    if(X11_HEADER)
        target_compile_definitions(real_engine PRIVATE RE_HAS_X11)
        find_library(X11_LIB X11 REQUIRED)
        target_link_libraries(real_engine PRIVATE ${X11_LIB})
    endif()
endif()

# GLSL -> SPIR-V (optional; skipped when glslangValidator is missing).
find_program(GLSLANG glslangValidator)
if(GLSLANG)
    set(SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SHADER_DIR})
    foreach(shader cull.comp pbr_clustered.vert pbr_clustered.frag shadow.vert tonemap.frag tonemap.vert)
        add_custom_command(
            OUTPUT ${SHADER_DIR}/${shader}.spv
            COMMAND ${GLSLANG} -V --target-env vulkan1.3
                    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}
                    -o ${SHADER_DIR}/${shader}.spv
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}
            COMMENT "Compiling ${shader}"
        )
        list(APPEND SPV_TARGETS ${SHADER_DIR}/${shader}.spv)
    endforeach()
    add_custom_target(shaders ALL DEPENDS ${SPV_TARGETS})
endif()

add_custom_target(verify
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/verify.sh
    DEPENDS real_engine
    COMMENT "Running formal verification suite"
)
