import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch-lbp-op',
    version='0.2',
    author="Patrick Knöbelreiter",
    author_email="knoebelreiter@icg.tugraz.at",
    packages=["src"],
    include_dirs=['../include/'],
    ext_modules=[
        CUDAExtension('pytorch_cuda_lbp_op', [
            'src/lbp.cpp',
            'src/lbp_min_sum_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        "numpy >= 1.15",
        "torch >= 0.4.1",
        "matplotlib >= 3.0.0",
        "scikit-image >= 0.14.1",
        "numba >= 0.42"
    ],
    )
