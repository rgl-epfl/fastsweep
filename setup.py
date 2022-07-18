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

setup(
    name="fastsweep",
    version="1.0.0",
    description="Eikonal solver using parallel fast sweeping",
    author="Delio Vicini",
    author_email="delio.vicini@gmail.com",
    license="BSD",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where="src"),
    install_requires=["drjit"],
    package_dir={"": "src"},
    cmake_install_dir="src/fastsweep",
    include_package_data=True,
    python_requires=">=3.8",
)
