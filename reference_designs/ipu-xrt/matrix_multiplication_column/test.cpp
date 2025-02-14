//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int m = 256;
constexpr int k = 128;
constexpr int n = 128;

constexpr int aVolume = m * k;
constexpr int bVolume = n * k;
constexpr int cVolume = m * n;

using A_DATATYPE = std::int16_t;
using B_DATATYPE = std::int16_t;
using C_DATATYPE = std::int16_t;

constexpr int aSize = (aVolume * sizeof(A_DATATYPE));
constexpr int bSize = (bVolume * sizeof(B_DATATYPE));
constexpr int cSize = (cVolume * sizeof(C_DATATYPE));

namespace po = boost::program_options;

void checkArgFileExists(po::variables_map &vmIn, std::string name) {
  if (!vmIn.count(name)) {
    throw std::runtime_error("Error: no " + name + " file was provided\n");
  }
  std::ifstream test(vmIn[name].as<std::string>());
  if (!test) {
    throw std::runtime_error("The " + name + " file " +
                             vmIn[name].as<std::string>() +
                             " does not exist.\n");
  }
}

std::vector<uint32_t> loadInstrSequence(std::string instr_path) {
  std::ifstream instrFile(instr_path);
  std::string line;
  std::vector<uint32_t> instrV;
  while (std::getline(instrFile, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instrV.push_back(a);
  }
  return instrV;
}

static inline std::int16_t randomInt16T() {
  return ((std::int16_t)rand() % 0x10000);
}

template <typename Tin, typename Tout>
void matmul(std::vector<Tin> a, std::vector<Tin> b, std::vector<Tout> &c) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      Tout runningSum = 0;
      for (int i = 0; i < k; i++) {
        runningSum += a[row * k + i] * b[i * n + col];
      }
      c[row * n + col] += runningSum;
    }
  }
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions to be sent to the LX6");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    return 1;
  }

  checkArgFileExists(vm, "xclbin");
  checkArgFileExists(vm, "instr");

  std::vector<uint32_t> instrV =
      loadInstrSequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instrV.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int deviceIndex = 0;
  auto device = xrt::device(deviceIndex);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto boInstr = xrt::bo(device, instrV.size() * sizeof(int),
                         XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto boA = xrt::bo(device, aSize, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto boB = xrt::bo(device, bSize, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto boC = xrt::bo(device, cSize, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";
  srand(static_cast<unsigned>(time(nullptr)));
  A_DATATYPE *bufA = boA.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> aVec;
  for (int i = 0; i < aVolume; i++)
    aVec.push_back(randomInt16T());
  memcpy(bufA, aVec.data(), (aVec.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB = boB.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> bVec;
  for (int i = 0; i < bVolume; i++)
    bVec.push_back(randomInt16T());
  memcpy(bufB, bVec.data(), (bVec.size() * sizeof(B_DATATYPE)));
  C_DATATYPE *bufC = boC.map<C_DATATYPE *>();
  std::vector<C_DATATYPE> cVec;
  for (int i = 0; i < cVolume; i++)
    cVec.push_back(0);
  memcpy(bufC, cVec.data(), (cVec.size() * sizeof(C_DATATYPE)));

  void *bufInstr = boInstr.map<void *>();
  memcpy(bufInstr, instrV.data(), instrV.size() * sizeof(int));

  boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto run = kernel(boInstr, instrV.size(), boA, boB, boC);
  run.wait();

  boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  C_DATATYPE *bufOut = boC.map<C_DATATYPE *>();

  int errors = 0;
  int maxErrors = 100;

  std::vector<C_DATATYPE> outputRef0;
  for (uint32_t i = 0; i < cVolume; i++)
    outputRef0.push_back(0);
  matmul(aVec, bVec, outputRef0);

  for (uint32_t i = 0; i < cVolume; i++) {
    if (bufOut[i] != outputRef0[i]) {
      errors++;
      if (errors < maxErrors) {
        std::cout << "\nerror, id " << i << " expected "
                  << std::to_string(outputRef0[i]) << ", got "
                  << std::to_string(bufOut[i]) << "\n";
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nerror count: " << errors << "\n\n";
  std::cout << "\nfailed.\n\n";
  return 1;
}
