# Install script for directory: /src/llama.cpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/src/build-cuda13/llama.cpp/ggml/src/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/src/build-cuda13/llama.cpp/ggml/src/libggml.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/src/llama.cpp/ggml/include/ggml.h"
    "/src/llama.cpp/ggml/include/ggml-cpu.h"
    "/src/llama.cpp/ggml/include/ggml-alloc.h"
    "/src/llama.cpp/ggml/include/ggml-backend.h"
    "/src/llama.cpp/ggml/include/ggml-blas.h"
    "/src/llama.cpp/ggml/include/ggml-cann.h"
    "/src/llama.cpp/ggml/include/ggml-cpp.h"
    "/src/llama.cpp/ggml/include/ggml-cuda.h"
    "/src/llama.cpp/ggml/include/ggml-opt.h"
    "/src/llama.cpp/ggml/include/ggml-metal.h"
    "/src/llama.cpp/ggml/include/ggml-rpc.h"
    "/src/llama.cpp/ggml/include/ggml-virtgpu.h"
    "/src/llama.cpp/ggml/include/ggml-sycl.h"
    "/src/llama.cpp/ggml/include/ggml-vulkan.h"
    "/src/llama.cpp/ggml/include/ggml-webgpu.h"
    "/src/llama.cpp/ggml/include/ggml-zendnn.h"
    "/src/llama.cpp/ggml/include/ggml-openvino.h"
    "/src/llama.cpp/ggml/include/gguf.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/src/build-cuda13/llama.cpp/ggml/src/libggml-base.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/src/build-cuda13/llama.cpp/ggml/ggml-config.cmake"
    "/src/build-cuda13/llama.cpp/ggml/ggml-config-version.cmake"
    )
endif()

