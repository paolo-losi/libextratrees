import glob
import numpy
from os.path import join
from setuptools import setup, find_packages, Extension

sources = [join("extratrees", "cbindings.c"),
           join("extratrees", "cextratrees.c")]

sources.extend(glob.glob(join("..", "src", "*.c")))

numpy_include = join(numpy.__path__[0], 'core', 'include')


setup(
    name='extratrees',
    version='dev',
    packages = find_packages(),
    ext_modules = [
        Extension("extratrees.cbindings", sources,
                  include_dirs=["../src", numpy_include],
                  extra_compile_args=["-std=c99", "-O2",
                                      "-Wno-unused", "-mtune=native"]),
                  #extra_compile_args=["-std=c99", "-Wno-unused"]),
    ],
)

