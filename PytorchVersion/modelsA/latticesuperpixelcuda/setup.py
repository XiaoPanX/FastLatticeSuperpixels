from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension 

setup(
    name='latticesuperpixel_cuda_cpp',
    ext_modules=[
        CUDAExtension('latticesuperpixel_cuda', [
            'latticesuperpixel_cuda.cpp',
            'latticesuperpixel_cuda_kernel.cu',
        ]),
       
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
