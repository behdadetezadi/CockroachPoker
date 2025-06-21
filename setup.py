#!/usr/bin/env python3
"""
Setup script for Cockroach Poker AI Bluff Detection Research Platform
"""

from setuptools import setup, find_packages
import os


# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


setup(
    name="cockroach-poker-ai",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="AI Bluff Detection Research Platform using Cockroach Poker",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/cockroach-poker-ai",
    project_urls={
        "Bug Reports": "https://github.com/username/cockroach-poker-ai/issues",
        "Source": "https://github.com/username/cockroach-poker-ai",
        "Documentation": "https://github.com/username/cockroach-poker-ai/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pygame>=2.5.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "analysis": [
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "cv": [
            "opencv-python>=4.8.0",
            "dlib>=19.24.0",
            "fer>=22.5.0",
            "mediapipe>=0.10.0",
        ],
        "ml": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-pygame>=1.0.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "all": [
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "opencv-python>=4.8.0",
            "dlib>=19.24.0",
            "fer>=22.5.0",
            "mediapipe>=0.10.0",
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "pytest>=7.4.0",
            "pytest-pygame>=1.0.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cockroach-poker=main:main",
            "cp-ai=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.py"],
        "": ["README.md", "requirements.txt", "LICENSE"],
    },
    keywords="ai artificial-intelligence bluff-detection game-theory pygame research machine-learning",
    platforms=["any"],
    zip_safe=False,
)

# Development installation helper
if __name__ == "__main__":
    import sys

    print("Cockroach Poker AI - Setup")
    print("=" * 40)

    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python setup.py install          # Install package")
        print("  python setup.py develop          # Development install")
        print("  pip install -e .                 # Editable install")
        print("  pip install -e .[all]            # Install with all extras")
        print("  pip install -e .[cv,analysis]    # Install with specific extras")
        print()
        print("Available extras:")
        print("  analysis   - Data analysis and visualization")
        print("  cv         - Computer vision for face detection")
        print("  ml         - Machine learning frameworks")
        print("  dev        - Development tools")
        print("  all        - All optional dependencies")
        sys.exit(0)