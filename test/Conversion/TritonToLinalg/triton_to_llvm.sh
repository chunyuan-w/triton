SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TRITON_BUILD_PYTHON39=temp.linux-x86_64-cpython-39
TRITON_BUILD_PYTHON38=temp.linux-x86_64-cpython-38
export PATH=$SCRIPT_DIR/../../../python/build/$TRITON_BUILD_PYTHON39/bin:$SCRIPT_DIR/../../../python/build/$TRITON_BUILD_PYTHON38/bin:$PATH
export PATH=~/.triton/llvm/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-18.04-release/bin:$PATH

triton-opt --triton-to-linalg $@ | mlir-opt  --mlir-print-ir-after-all -empty-tensor-to-alloc-tensor -one-shot-bufferize="allow-return-allocs" --test-lower-to-llvm | mlir-translate --mlir-to-llvmir

