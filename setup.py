from setuptools import setup, find_packages

setup(
    name="sfnet",
    version="1.0.0",
    description="A neural network-based frame interpolation system",
    author="Hamid Azadegan",
    author_email="hamidazad2001@example.com",
    url="https://github.com/hamidazad2001/SF_Net-v1",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.4.0",
        "gin-config",
        "absl-py",
        "tensorflow-addons",
        "numpy"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 