from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["*.pyx"],
        include_dirs = [numpy.get_include(), "include", "src"],
        libraries = [],
        library_dirs = [],
        extra_compile_args = ['-fopenmp', '-std=c++11'],
        extra_link_args = ['-fopenmp']),
]
setup(
    name = "FTPRL",
    ext_modules = cythonize(extensions),
    install_requires = [
      "numpy",
      "scipy"
    ],
)
