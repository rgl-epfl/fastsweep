import sys, os

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

fastsweep_cmake_toolchain_file = os.environ.get("FASTSWEEP_CMAKE_TOOLCHAIN_FILE", "")
fastsweep_drjit_cmake_dir = os.environ.get("FASTSWEEP_DRJIT_CMAKE_DIR", "")

setup(
    name="fastsweep",
    version="0.1.0",
    description="Eikonal solver using parallel fast sweeping",
    author="Delio Vicini",
    author_email="delio.vicini@gmail.com",
    license="BSD",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where="src"),
    install_requires=["drjit"],
    package_dir={"": "src"},
    cmake_args=[
        f'-DCMAKE_TOOLCHAIN_FILE={fastsweep_cmake_toolchain_file}',
        f'-DDRJIT_CMAKE_DIR:STRING={fastsweep_drjit_cmake_dir}'
    ],

    cmake_install_dir="src/fastsweep",
    include_package_data=True,
    python_requires=">=3.8",
)
