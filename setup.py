# D:\ZhaoyangProject\DVisionix\setup.py

"""DVisionix 安装脚本。"""

import os

from setuptools import setup, find_packages


def _read_long_description() -> str:
    if os.path.exists("README.md"):
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="dvisionix",
    version="0.14.0",
    author="DVisionix Team",
    description="A PyTorch-based deep learning training framework for computer vision tasks",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0,<3",
        "torchvision>=0.15",
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
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "timm>=0.14.0",
            "albumentations>=1.3.0",
            "onnx>=1.13.0",
            "onnxruntime>=1.14.0",
            "pycocotools>=2.0.6",
            "torchmetrics>=1.0.0",
        ],
    },
    python_requires=">=3.10",
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
