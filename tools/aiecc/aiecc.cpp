#include <iostream>
#include <string>
#include <regex>
#include <stdlib.h>
#include <filesystem>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/SourceMgr.h"
#include "aie/InitialAllDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "aie/Targets/AIETargets.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "configure.h"

using namespace llvm;
using namespace mlir;
namespace sfs = std::filesystem;

cl::opt<std::string> FileName(cl::Positional, cl::desc("<input mlir>"), cl::Required);
cl::opt<std::string> TmpDir("tmpdir", cl::desc("Directory used for temporary file storage"));
cl::opt<std::string> SysRoot("sysroot", cl::desc("sysroot for cross-compilation"));
cl::opt<bool> Verbose("v", cl::desc("Trace commands as they are executed"));
cl::opt<bool> Vectorize("vectorize", cl::desc("Enable MLIR vectorization"));
// NOTE: requires chess
//cl::opt<bool> AIESim("aiesim", cl::desc("Generate aiesim Work folder"));
cl::opt<std::string> Peano("peano", cl::desc("Root directory where peano compiler is installed"));
cl::opt<bool> Compile("compile", cl::desc("Enable compiling of AIE code"), cl::init(true));
cl::opt<bool> Link("link", cl::desc("Enable compiling of AIE code"), cl::init(true));
cl::opt<std::string> HostArch("host-target", cl::desc("Target architecture of the host program"), cl::init(HOST_ARCHITECTURE));
cl::opt<bool> HostCompile("compile-host", cl::desc("Enable compiling of the host program"), cl::init(true));
cl::list<std::string> HostArgs(cl::Positional, cl::desc("[<host compiler arg>]..."));
cl::opt<unsigned> Threads("j", cl::desc("Compile with max n-threads in the machine (default is 4).  An argument of zero corresponds to the maximum number of threads on the machine."), cl::init(4));
cl::opt<bool> Profiling("profile", cl::desc("Profile commands to find the most expensive executions."));
cl::opt<bool> Unified("unified", cl::desc("Compile all cores together in a single process"));
cl::opt<bool> DryRun("n", cl::desc("Disable actually executing any commands."));
cl::opt<bool> Progress("progress", cl::desc("Show progress visualization."));
cl::opt<bool> GenerateIPU("aie-generate-ipu", cl::desc("Generate ipu instruction stream"));
cl::opt<bool> OnlyGenerateIPU("aie-only-generate-ipu", cl::desc("Generate ipu instruction stream only"));
cl::opt<std::string> IPUInstsName("ipu-insts-name", cl::desc("Output instructions filename for IPU target"), cl::init("ipu_insts.txt"));
cl::opt<std::string> XCLBinName("xclbin-name", cl::desc("Output xclbin filename for CDO/XCLBIN target"), cl::init("final.xclbin"));
cl::opt<std::string> XCLBinKernelName("xclbin-kernel-name", cl::desc("Kernel name in xclbin file"), cl::init("MLIR_AIE"));
cl::opt<std::string> XCLBinInstanceName("xclbin-instance-name", cl::desc("Instance name in xclbin metadata"), cl::init("MLIRAIE"));
cl::opt<std::string> XCLBinKernelID("xclbin-kernel-id", cl::desc("Kernel id in xclbin file"), cl::init("0x901"));

void addLowerToLLVMPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass({.useBarePtrCallConv=true}));
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

int runTool(StringRef Program, ArrayRef<StringRef> Args) {
  if (Verbose) {
    llvm::outs() << "Run: " << Program;
    for (auto &s : Args) {
      llvm::outs() << " " << s;
    }
    llvm::outs() << "\n";
  }
  std::string err_msg;
  sys::ProcessStatistics stats;
  std::optional<sys::ProcessStatistics> opt_stats(stats);
  SmallVector<StringRef, 8> PArgs = {Program};
  PArgs.append(Args.begin(), Args.end());
  int result = sys::ExecuteAndWait(Program, PArgs, std::nullopt, {}, 0, 0, &err_msg, nullptr, &opt_stats);
  if (Verbose) {
    llvm::outs() << (result == 0 ? "Succeeded " : "Failed ") << "in " <<
      std::chrono::duration_cast<std::chrono::duration<float>>(stats.TotalTime).count() << " code: " << result << "\n";
  }
  return result;
}

template <unsigned N>
void aieTargetDefines(SmallVector<StringRef, N> &Args, std::string aie_target) {
  if (aie_target == "AIE2") {
    Args.push_back("-D__AIEARCH__=20");
  } else {
    Args.push_back("-D__AIEARCH__=10");
  }
}

int main(int argc, char *argv[])
{
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerTranslationCLOptions();
  cl::ParseCommandLineOptions(argc, argv);

  const char * env_vitis = ::getenv("VITIS");
  sfs::path aietools_path;
  if (env_vitis == nullptr) {
    if (auto vpp = sys::findProgramByName("v++")) {
      SmallString<50> real_vpp;
      std::error_code err = sys::fs::real_path(vpp.get(), real_vpp);
      if (!err) {
        sys::path::remove_filename(real_vpp);
        sys::path::remove_filename(real_vpp);
        ::setenv("VITIS", real_vpp.c_str(), 1);
        std::cout << "Found Vitis at " << real_vpp.c_str() << std::endl;
      }
    }
  }
  env_vitis = ::getenv("VITIS");
  if (env_vitis != nullptr) {
    sfs::path vitis_path(env_vitis);
    sfs::path vitis_bin_path = vitis_path / "bin";

    aietools_path = vitis_path / "aietools";
    if (!sfs::exists(aietools_path)) {
      aietools_path = vitis_path / "cardano";
    }
    ::setenv("AIETOOLS", aietools_path.c_str(), 1);

    sfs::path aietools_bin_path = aietools_path / "bin";
    const char * env_path = ::getenv("PATH");
    if (env_path == nullptr)
      env_path = "";
    std::string new_path(env_path);
    if (new_path.size())
      new_path += sys::EnvPathSeparator;
    new_path += aietools_bin_path.c_str();
    new_path += sys::EnvPathSeparator;
    new_path += vitis_bin_path.c_str();
    ::setenv("PATH", new_path.c_str(), 1);
  } else {
    std::cout << "VITIS not found ..." << std::endl;
  }

  if (Verbose) {
    std::cout << std::endl << "Compliing " << FileName << std::endl;
  }

  sfs::path temp_dir;
  if (TmpDir.size()) {
    temp_dir = TmpDir.getValue();
  } else {
    temp_dir = FileName + ".prj";
  }
  temp_dir = sfs::absolute(temp_dir);

  std::error_code err;
  sfs::create_directory(temp_dir, err);
  if (err) {
    std::cerr << "Failed to create temporary directory " << temp_dir << ": " << err.message() << std::endl;
    std::exit(1);
  }

  if (Verbose) {
    std::cout << "Created temporary directory " << temp_dir << std::endl;
  }

  MLIRContext ctx;
  ParserConfig pcfg(&ctx);
  SourceMgr srcMgr;

  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  xilinx::registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> owning =
      parseSourceFile<ModuleOp>(FileName, srcMgr, pcfg);

  if (!owning) {
    return 1;
  }

  owning->dump();

  PassManager pm(&ctx, owning->getOperationName());
  pm.addPass(createLowerAffinePass());
  pm.addPass(xilinx::AIE::createAIECanonicalizeDevicePass());
  OpPassManager &devicePM = pm.nest<xilinx::AIE::DeviceOp>();
  devicePM.addPass(xilinx::AIE::createAIEAssignLockIDsPass());
  devicePM.addPass(xilinx::AIE::createAIEObjectFifoRegisterProcessPass());
  devicePM.addPass(xilinx::AIE::createAIEObjectFifoStatefulTransformPass());
  devicePM.addPass(xilinx::AIEX::createAIEBroadcastPacketPass());
  devicePM.addPass(xilinx::AIE::createAIERoutePacketFlowsPass());
  devicePM.addPass(xilinx::AIEX::createAIELowerMulticastPass());
  devicePM.addPass(xilinx::AIE::createAIEAssignBufferAddressesPass());
  pm.addPass(createConvertSCFToCFPass());

  if (Verbose) {
    llvm::outs() << "Running: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  if (failed(pm.run(*owning))) {
    return 1;
  }

  owning->dump();

  std::string target_arch;
  raw_string_ostream target_arch_os(target_arch);
  if (failed(xilinx::AIE::AIETranslateToTargetArch(*owning, target_arch_os))) {
    return 1;
  }

  target_arch = StringRef(target_arch).trim();

  std::regex target_regex("AIE.?");
  if (!std::regex_search(target_arch, target_regex)) {
    std::cerr << "Unexpected target architecture: " << target_arch << std::endl;
    return 1;
  }

  std::string peano_target = StringRef(target_arch).lower() + "-none-elf";

  std::cout << "target arch: " << peano_target << std::endl;

  if (GenerateIPU || OnlyGenerateIPU) {
    PassManager pm(&ctx, owning->getOperationName());
    pm.addNestedPass<xilinx::AIE::DeviceOp>(xilinx::AIEX::createAIEDmaToIpuPass());
    ModuleOp copy = owning->clone();
    if (failed(pm.run(copy))) {
      return 1;
    }
    
    std::string errorMessage;
    auto output = openOutputFile(IPUInstsName, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }

    if (failed(xilinx::AIE::AIETranslateToIPU(copy, output->os()))) {
      return 1;
    }

    output->keep();

    if (OnlyGenerateIPU) {
      return 0;
    }
  }

  sfs::path unified_obj = temp_dir / "input.o";

  sfs::path peano_bin(Peano.getValue());
  sfs::path peano_opt = peano_bin / "opt";
  sfs::path peano_llc = peano_bin / "llc";
  sfs::path install_dir = sfs::absolute(argv[0]).parent_path().parent_path();

  if (Unified) {
    PassManager pm(&ctx, owning->getOperationName());
    pm.addNestedPass<xilinx::AIE::DeviceOp>(xilinx::AIE::createAIELocalizeLocksPass());
    pm.addNestedPass<xilinx::AIE::DeviceOp>(xilinx::AIE::createAIENormalizeAddressSpacesPass());
    pm.addPass(xilinx::AIE::createAIECoreToStandardPass());
    pm.addPass(xilinx::AIEX::createAIEXToStandardPass());
    addLowerToLLVMPasses(pm);

    ModuleOp copy = owning->clone();
    if (failed(pm.run(copy))) {
      return 1;
    }

    sfs::path file_llvmir = temp_dir / "input.ll";

    std::string errorMessage;
    auto output = openOutputFile(file_llvmir.string(), &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(copy, llvmContext);
    if (!llvmModule)
      return 1;

    llvmModule->print(output->os(), nullptr);
    output->keep();
    

    if (Compile) {
      sfs::path file_llvmir_opt = temp_dir / "input.opt.ll";
      if(runTool(peano_opt.string(), {"--passes=default<O2>", "-inline-threshold=10", "-S", file_llvmir.string(), "-o", file_llvmir_opt.string()}) != 0) {
        return 1;
      }
      if(runTool(peano_llc.string(), {file_llvmir_opt.string(), "-O2", "--march=" + StringRef(target_arch).lower(), "--function-sections", "--filetype=obj", "-o", unified_obj.string()}) != 0) {
        return 1;
      }
    }
  }

  // host generation
  {
    if (Verbose) {
      std::cout << "Host compilation\n";
    }
    ModuleOp copy = owning->clone();
    PassManager pm(&ctx, owning->getOperationName());
    pm.addNestedPass<xilinx::AIE::DeviceOp>(xilinx::AIE::createAIEPathfinderPass());
    pm.addNestedPass<xilinx::AIE::DeviceOp>(xilinx::AIEX::createAIEBroadcastPacketPass());
    pm.addNestedPass<xilinx::AIE::DeviceOp>(xilinx::AIE::createAIERoutePacketFlowsPass());
    pm.addNestedPass<xilinx::AIE::DeviceOp>(xilinx::AIEX::createAIELowerMulticastPass());

    if (failed(pm.run(copy))) {
      return 1;
    }
    sfs::path file_inc_cpp = temp_dir / "aie_inc.cpp";

    std::string errorMessage;
    auto out_inc_cpp = openOutputFile(file_inc_cpp.string(), &errorMessage);
    if (!out_inc_cpp) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }

    if (failed(xilinx::AIE::AIETranslateToXAIEV2(copy, out_inc_cpp->os()))) {
      return 1;
    }

    sfs::path file_ctrl_cpp = temp_dir / "aie_control.cpp";

    auto out_ctrl_cpp = openOutputFile(file_ctrl_cpp.string(), &errorMessage);
    if (!out_ctrl_cpp) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }

    if (failed(xilinx::AIE::AIETranslateToCDO(copy, out_ctrl_cpp->os()))) {
      return 1;
    }

    out_inc_cpp->keep();
    out_ctrl_cpp->keep();

    if (HostArgs.size() > 0) {
      SmallVector<StringRef, 20> ClangArgs = {"-std=c++11"};
      if (HostArch.size()) {
        ClangArgs.push_back("--target=" + HostArch);
      }

      if (SysRoot.size()) {
        ClangArgs.push_back("--sysroot=" + SysRoot);
        /* In order to find the toolchain in the sysroot, we need to have
          a 'target' that includes 'linux' and for the 'lib/gcc/$target/$version'
          directory to have a corresponding 'include/gcc/$target/$version'.
          In some of our sysroots, it seems that we find a lib/gcc, but it
          doesn't have a corresponding include/gcc directory.  Instead
          force using '/usr/lib,include/gcc' */
        if (HostArch == "aarch64-linux-gnu") {
          ClangArgs.push_back("--gcc-toolchain=" + SysRoot + "/usr");
        }
      }
      size_t dash = HostArch.find('-');
      if (dash == std::string::npos)
        dash = HostArch.size();
      sfs::path runtime_path = install_dir / "runtime_lib" / HostArch.substr(0, dash);
      sfs::path xaiengine_include =  runtime_path / "xaiengine" / "include";
      sfs::path xaiengine_lib = runtime_path / "xaiengine" / "lib";
      sfs::path runtime_testlib = runtime_path / "test_lib" / "lib" / "libmemory_allocator_ion.a";

      ClangArgs.append({runtime_testlib.string(),
      "-I" + xaiengine_include.string(),
      "-L" + xaiengine_lib.string(),
      "-L" + (aietools_path / "lib" / "lnx64.so").string(),
      "-I" + temp_dir.string(),
      "-fuse-ld=lld",
      "-lm",
      "-lxaiengine"});

      aieTargetDefines(ClangArgs, target_arch);

      ClangArgs.append(HostArgs.begin(), HostArgs.end());

      runTool("clang++", ClangArgs);
    }
  }
  
  return 0;
}
