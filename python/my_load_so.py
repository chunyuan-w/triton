# get .o from llvm emit
# get .so from clang
# clang  tmp_obj.o  -shared -o tmp_obj.so
# use the below code to load .so and run the function

from ctypes import cdll
import torch 

path = "/home/eikan/local_disk/chunyuan/inductor/tmp_obj.so"
lib = cdll.LoadLibrary(path)
print("lib:", lib.add_kernel_0d1d2c)

kernel = lib.add_kernel_0d1d2c

kernel(torch.randn(1), torch.randn(2))

