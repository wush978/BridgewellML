from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["src/*.pyx"],
        include_dirs = [numpy.get_include(), "include", "src"],
        libraries = [],
        library_dirs = [],
        extra_compile_args = ['-fopenmp', '-std=c++11'],
        extra_link_args = ['-fopenmp']),
]

#with open("CHANGELOG", "r") as f:
#    version = f.readline().rstrip()
version = "0.1.1"

setup(
    name = "FTPRL",
    version = version,
    ext_modules = cythonize(extensions),
    install_requires = [
      "cython",
      "numpy",
      "scipy"
    ],
)
