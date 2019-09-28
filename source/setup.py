# distutils: language_level = 3
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["*.pyx"])
)