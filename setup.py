# D:\ZhaoyangProject\DVisionix\setup.py

"""
DVisionix 安装脚本
"""

from setuptools import setup, find_packages

setup(
    name="dvisionix",
    version="0.1.0",
    author="DVisionix Team",
    description="A PyTorch-based deep learning training framework for computer vision tasks",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.12.0",
        "torchvision>=0.27.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "tensorboard>=2.11.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "timm>=0.9.0",
            "albumentations>=1.3.0",
            "onnx>=1.13.0",
            "onnxruntime>=1.14.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)