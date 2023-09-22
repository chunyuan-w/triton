#include "triton/Target/PTX/PTXTranslation.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include <filesystem>

#include <mutex>
#include <optional>
#include <fstream>

#include <mutex>
#include <optional>

namespace triton {

static void initLLVM() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

static bool findAndReplace(std::string &str, const std::string &begin,
                           const std::string &end, const std::string &target) {
  size_t startReplace = str.find(begin);
  if (startReplace == std::string::npos)
    return false;
  size_t endReplace = str.find(end, startReplace);
  if (endReplace == std::string::npos)
    return false;
  str.replace(startReplace, endReplace + 1 - startReplace, target);
  return true;
}

std::string translateLLVMIRToPTX(llvm::Module &module, int cc, int version) {
  // LLVM version in use may not officially support target hardware.
  // Supported versions for LLVM 14 are here:
  // https://github.com/llvm/llvm-project/blob/f28c006a5895fc0e329fe15fead81e37457cb1d1/clang/include/clang/Basic/BuiltinsNVPTX.def
  int maxPTX = std::min(82, version);
  int maxCC = std::min(90, cc);
  // options
  auto options = llvm::cl::getRegisteredOptions();
  auto *shortPtr =
      static_cast<llvm::cl::opt<bool> *>(options["nvptx-short-ptr"]);
  assert(shortPtr);
  shortPtr->setValue(true);
  std::string sm = cc == 90 ? "sm_90a" : "sm_" + std::to_string(cc);
  // max PTX version
  int ptxMajor = maxPTX / 10;
  int ptxMinor = maxPTX % 10;
  // create
  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = "sm_" + std::to_string(maxCC);
  std::string layout = "";
  std::string features = "";
  // std::string features = "+ptx" + std::to_string(maxPTX);
  for (llvm::Function &f : module.functions()) {
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  }
  initLLVM();
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(triple);
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(
      module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOpt::Aggressive);
  // set data layout
  if (layout.empty())
    module.setDataLayout(machine->createDataLayout());
  else
    module.setDataLayout(layout);
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    for (llvm::Function &f : module.functions())
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::legacy::PassManager pass;
    // emit
    machine->addPassesToEmitFile(pass, pstream, nullptr,
                                 llvm::CodeGenFileType::CGFT_AssemblyFile);
    pass.run(module);
  }
  // post-process
  findAndReplace(result, ".version", "\n",
                 ".version " + std::to_string(ptxMajor) + "." +
                     std::to_string(ptxMinor) + "\n");
  findAndReplace(result, ".target", "\n", ".target " + sm + "\n");
  while (findAndReplace(result, "\t// begin inline asm", "\n", ""))
    ;
  while (findAndReplace(result, "\t// end inline asm", "\n", ""))
    ;
  return result;
}


std::string translateLLVMIRToASM(llvm::Module &module, int cc, int version) {
    printf("before InitializeAllTargets\n");
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    // llvm::InitializeAllAsmParsers();
    printf("after InitializeAllTargets\n");

    // Get the default target triple.
    std::string defaultTargetTriple = llvm::sys::getDefaultTargetTriple();

    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(defaultTargetTriple, error);
    if (!target) {
        llvm::errs() << "Failed to initialize target: " << error << '\n';
        return "";
    }
    printf("before targetMachine\n");
    llvm::TargetOptions targetOptions;
    llvm::CodeGenOpt::Level optimizationLevel = llvm::CodeGenOpt::Default;
    printf("before createTargetMachine\n");
    llvm::errs() << "defaultTargetTriple" << defaultTargetTriple << "\n";
    llvm::errs() << "getHostCPUName" << llvm::sys::getHostCPUName() << "\n";
    
    llvm::TargetMachine *targetMachine = target->createTargetMachine(defaultTargetTriple,
                                                                    llvm::sys::getHostCPUName(),
                                                                    "",
                                                                    targetOptions,
                                                                    llvm::Reloc::PIC_,
                                                                    llvm::CodeModel::Kernel,
                                                                    optimizationLevel);
    printf("after createTargetMachine\n");

    if (!targetMachine) {
    printf("fail createTargetMachine\n");

        llvm::errs() << "Failed to create target machine\n";
        return "";
    }
    printf("before stream\n");

    
    llvm::SmallString<64> fsrc;
    llvm::sys::fs::createTemporaryFile("compile-ptx-src", "", fsrc);
    std::string fbin = std::string(fsrc) + ".o";
    llvm::FileRemover binRemover(fbin);
    const char *_fsrc = fsrc.c_str();
    const char *_fbin = fbin.c_str();
    auto outputFileName = std::string(_fsrc) + ".o";
    
    std::string cubin;

    // Set up a raw string stream to capture the assembly output.
    std::string result;
    {
      llvm::raw_string_ostream stream(result);
      llvm::buffer_ostream pstream(stream);


      // std::string outputFileName = "/home/eikan/local_disk/chunyuan/inductor/tmp_obj.o";
      std::error_code error;
      llvm::raw_fd_ostream dest(outputFileName, error);
    printf("before addPassesToEmitFile\n");

      // Create a PassManager and add the target-specific code generator pass.
      llvm::legacy::PassManager passManager;
      targetMachine->addPassesToEmitFile(passManager, dest, nullptr, llvm::CodeGenFileType::CGFT_ObjectFile);

      // Run the optimization and code generation passes on the module.
      if (llvm::verifyModule(module)) {
          llvm::errs() << "Module verification failed\n";
          return "";
      }
      printf("before passManager\n");

      passManager.run(module);







      llvm::FileRemover srcRemover(fsrc);
      std::ifstream _cubin(_fbin, std::ios::binary);
      cubin = std::string(std::istreambuf_iterator<char>(_cubin), {});
      _cubin.close();



    }
    // Clean up the target machine.
    delete targetMachine;
    printf("before result\n");

      return cubin;
}


} // namespace triton
