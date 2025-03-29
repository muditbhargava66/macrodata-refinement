"""
Setup script for the Macrodata Refinement (MDR) package.
"""

import os
import re
from typing import List, Dict, Any, Optional
from setuptools import setup, find_packages


def assert_is_string(value: Any, name: str) -> None:
    """
    Assert that a value is a string.
    
    Args:
        value: Value to check
        name: Name of the value for error message
    """
    assert isinstance(value, str), f"{name} must be a string"


def get_version() -> str:
    """
    Get the version from the package __init__.py.
    
    Returns:
        Version string
    """
    init_path = os.path.join("src", "mdr", "__init__.py")
    
    # If the file doesn't exist, return a default version
    if not os.path.exists(init_path):
        return "0.1.0"
    
    with open(init_path, "r", encoding="utf-8") as f:
        init_contents = f.read()
    
    # Extract version using regex
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_contents)
    
    if version_match:
        version = version_match.group(1)
        assert_is_string(version, "version")
        return version
    else:
        return "0.1.0"


def get_long_description() -> str:
    """
    Get the long description from the README.md file.
    
    Returns:
        Long description string
    """
    # If the README file doesn't exist, return a default description
    if not os.path.exists("README.md"):
        return "Macrodata Refinement (MDR) - A toolkit for refining and processing macrodata."
    
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    
    assert_is_string(long_description, "long_description")
    return long_description


def get_requirements() -> List[str]:
    """
    Get requirements from the requirements.txt file.
    
    Returns:
        List of requirements
    """
    # If the requirements file doesn't exist, return a default list
    if not os.path.exists("requirements.txt"):
        return [
            "numpy>=1.23.0",
            "pandas>=1.5.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "pyarrow>=10.0.0",
            "h5py>=3.7.0",
            "psutil>=5.9.0",
        ]
    
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # Validate that each requirement is a string
    for req in requirements:
        assert_is_string(req, "requirement")
    
    return requirements


# Get package metadata
version = get_version()
long_description = get_long_description()
requirements = get_requirements()

# Additional dependencies for development
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "mypy>=0.990",
    "flake8>=6.0.0",
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "pre-commit>=2.20.0",
    "tabulate>=0.9.0",  # For benchmark script
]

# Additional dependencies for documentation
docs_requirements = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.19.0",
    "nbsphinx>=0.8.9",
    "ipython>=8.0.0",
]

# Run setup
setup(
    name="macrodata-refinement",
    version=version,
    description="A toolkit for refining, validating, and transforming macrodata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mudit Bhargava",
    author_email="muditbhargava666@gmail.com",
    url="https://github.com/muditbhargava66/macrodata-refinement",
    project_urls={
        "Bug Tracker": "https://github.com/muditbhargava66/macrodata-refinement/issues",
        "Documentation": "https://github.com/muditbhargava66/macrodata-refinement/docs",
        "Source Code": "https://github.com/muditbhargava66/macrodata-refinement",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
    },
    entry_points={
        "console_scripts": [
            "mdr=mdr.api.cli:main",
        ],
    },
)