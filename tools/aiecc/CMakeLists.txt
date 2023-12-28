#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

cmake_minimum_required(VERSION 3.21)
project(aiecc)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(_aiecc_srcs aiecc.cpp)

add_executable(aiecc ${_aiecc_srcs})

target_include_directories(aiecc PUBLIC ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_update_compile_flags(aiecc)

llvm_map_components_to_libnames(llvm_libs support)
target_link_libraries(aiecc ${llvm_libs})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

configure_file(configure.h.in configure.h)
target_include_directories(aiecc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(aiecc
  ${dialect_libs}
  MLIRParser
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  ADF
  AIE
  AIETransforms
  AIETargets
  AIEX
  AIEXTransforms
  MLIRAIEVec)

install(TARGETS aiecc
  EXPORT AIECC
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT aiecc)
