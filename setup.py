from setuptools import setup, find_packages
import os
import subprocess
import platform


def is_linux():
    print("Platform system: ",platform.system())
    return platform.system() == 'Linux'

if is_linux():
    print("The system is running on Linux.")
    system = 0
    
else:
    print("The system is not running on Linux.")
    system = 1

def install_dependencies():
    torch_dependencies = "https://download.pytorch.org/whl/torch_stable.html"

    if system == 0:
        subprocess.check_call(["pip", "install", "torch==2.3.1+cu118", "torchvision==0.18.1+cu118", "-f", torch_dependencies])
    else:
        subprocess.check_call(["pip", "install", "torch==2.3.1", "torchvision==0.18.1", "-f", torch_dependencies])
        
    subprocess.check_call(["pip", "install", "git+https://github.com/Rich-Hall/sentinel1decoder"])
    
    with open("requirements.txt", "r") as f:
        install_requires = []
        packages = f.read().split("\n")
        for pack in packages:
            if pack != "":
                package = pack.replace(' ','')
                install_requires.append(package)
                subprocess.check_call(["pip", "install", package])
                
        return install_requires

def long_description_reader():
    with open("README.md", encoding="utf-8") as fh:
        long_description = fh.read()
    return long_description

install_dependencies()

LD = long_description_reader()

setup(
    name='SARLens',
    version='0.1',
    description='SAR Focusing using AI',
    long_description=LD,
    long_description_content_type="text/markdown",
    author='Roberto Del Prete',
    author_email='roberto.delprete@ext.esa.int',
    packages=find_packages(),
    install_requires=["asf_search",
                        "pandas",
                        "setuptools-git",
                        "rasterio",
                        "matplotlib",
                        "torch",
                        "torchvision",
                        "torchmetrics",
                        "seaborn",
                        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires=">=3.10, <4",
    project_urls={"Source": "https://github.com/sirbastiano/AutoFocusNet"},
)