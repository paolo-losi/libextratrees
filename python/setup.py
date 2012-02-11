import glob
import numpy
from os.path import join
from setuptools import setup, find_packages, Extension

sources = [join("src", "cextratrees.c"),
           join("src", "extratrees.c")]

sources.extend(glob.glob(join("..", "src", "*.c")))

numpy_include = join(numpy.__path__[0], 'core', 'include')


setup(
    name='extratrees',
    version='dev',
    packages = find_packages(),
    ext_modules = [
        Extension("extratrees", sources,
                  include_dirs=["../src", numpy_include],
                  extra_compile_args=["-std=c99"]),
    ],
)

