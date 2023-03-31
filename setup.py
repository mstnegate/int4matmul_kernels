from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='int4matmul',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'int4matmul',
            [
                'int4matmul.cpp',
                'matmul_kernel.cu',
                'matvec_kernel.cu',
                'utils_kernel.cu',
            ],
            extra_compile_args={'nvcc': ['-O2', '--generate-line-info']},
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)