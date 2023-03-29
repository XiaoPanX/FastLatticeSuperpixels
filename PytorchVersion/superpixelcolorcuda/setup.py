from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension 

setup(
    name='superpixelcolor_cuda_cpp',
    ext_modules=[
        CUDAExtension('superpixelcolor_cuda', [
            'superpixelcolor_cuda.cpp',
            'superpixelcolor_cuda_kernel.cu',
        ]),
       
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
