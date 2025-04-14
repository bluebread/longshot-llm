# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.1.0"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("longshot._core",
        ["src/longshot/core/core.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [
            ('VERSION_INFO', __version__)
        ],
        cxx_std = 17,
        include_dirs = ["src/longshot/core"],
        extra_compile_args = [
            "-Ofast", "-fopenmp"
        ]
        ),
]

setup(
    name="gym-longshot",
    version=__version__,
    author="bluebread",
    author_email="hotbread70127@gmail.com",
    url="https://github.com/bluebread/gym-longshot",
    description="A C++/Python library for boolean function",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    # # Currently, build_ext only provides an optional "highest supported C++
    # # level" feature, but in the future it may provide more features.
    # cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)