from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crossentropy-triton",
    version="0.1.1",
    author="Daniel Castillo",
    author_email="d.castillocastagneto@gmail.com",
    description="A high-performance, memory-efficient cross-entropy loss implementation using Triton for CUDA GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dcas89/crossentropy-triton",
    project_urls={
        "Bug Reports": "https://github.com/Dcas89/crossentropy-triton/issues",
        "Source": "https://github.com/Dcas89/crossentropy-triton",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.6.0",
        "triton>=3.3.0",
    ],
    keywords="triton, cuda, cross-entropy, machine-learning, deep-learning, pytorch",
)
