# Computing_on-GPU
GENERAL PURPOSE COMPUTING ON GRAPHICS PROCESSING UNITS

A GPU (graphics processing unit) is a computer chip that generates graphics
and images by performing fast mathematical computations (Kruglov et al., 2016)
The purpose of GPGPUs are to grasp the power of GPUs and to carry out tasks
previously done by central processing units (CPUs)
CUDA GPUs are good at performing arithmetic operations but face challenges
with memory access
Memory-bound kernels spend a lot of time reading from and writing to global
memory, leading to performance bottlenecks
To address this issue, in this study the researchers aim to automate the fusion of
kernels (through source-to-source compilers) to reduce memory transfers and
enhance overall performance of GPU computations (Filipovič et al., 2015)

It is hypothesised that automated kernel fusion with specific characteristics such as
kernels that perform map and reduce operations can increase the efficiency and
enhance performance of the GPU memory-bound computations. A proof of concept is provided 
"POC.cu" to test this hypothesis
