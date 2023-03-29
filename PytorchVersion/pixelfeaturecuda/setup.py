from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension 

setup(
    name='pixelfeature_cuda_cpp',
    ext_modules=[
        CUDAExtension('pixelfeature_cuda', [
            'pixelfeature_cuda.cpp',
            'pixelfeature_cuda_kernel.cu',
        ]),
       
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
