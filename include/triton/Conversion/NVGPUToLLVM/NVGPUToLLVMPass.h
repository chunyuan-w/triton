#ifndef TRITON_CONVERSION_NVGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_NVGPU_TO_LLVM_PASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUToLLVMPass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertCPUToLLVMPass();

} // namespace triton

} // namespace mlir

#endif
