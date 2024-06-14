from setuptools import setup, find_packages
import os
import subprocess

setup(
    name='SARLens',
    version='0.1',
    description='SAR Focusing using AI',
    long_description='A python package for SAR focusing and AI designed with torch and lightning. Sentinel-1 decoeder imported from Rich-Hall',
    long_description_content_type="text/markdown",
    author='Roberto Del Prete',
    author_email='roberto.delprete@ext.esa.int',
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.rst", "*.md"],
    },
    install_requires=["asf_search==7.1.2",
                        "pandas",
                        "rasterio",
                        "matplotlib",
                        "torch",
                        "torchvision",
                        "torchmetrics",
                        "seaborn",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    packages=["SARLens", "SARLens.autofocus", "SARLens.processor", "SARLens.utils", "sentinel1decoder"],
    python_requires=">=3.10, <4",
    project_urls={"Source": "https://github.com/sirbastiano/AutoFocusNet"},
)