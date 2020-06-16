import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matching',
    version='1.0',
    author="Patrick Knöbelreiter",
    author_email="knoebelreiter@icg.tugraz.at",
    packages=["sad"],
    include_dirs=['include/' ],
    ext_modules=[
        CUDAExtension('pytorch_cuda_stereo_sad_op', [
            'sad/src/stereo_sad.cpp',
            'sad/src/stereo_sad_kernel.cu',
        ]),
        CUDAExtension('pytorch_cuda_flow_mp_sad_op', [
            'flow_mp_sad/src/flow_mp_sad.cpp',
            'flow_mp_sad/src/flow_mp_sad_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        "numpy >= 1.15",
        "torch >= 0.4.1",
        "scikit-image >= 0.14.1",
    ],
    )
