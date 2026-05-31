"""
DREDGE Studio - Advanced Security & Intelligence Platform
Setup configuration for Python package distribution
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements = [
    "flask>=2.0.0",
    "requests>=2.25.0",
    "numpy>=1.19.0",
    "torch>=2.0.0",
]

# Development requirements
dev_requirements = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "black>=21.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
]

setup(
    name="dredge-studio",
    version="2.0.0",
    author="DREDGE Team",
    author_email="team@dredge.dev",
    description="Advanced Security Intelligence and Model Management Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/docker/dredge-cli-repo",
    project_urls={
        "Bug Tracker": "https://github.com/docker/dredge-cli-repo/issues",
        "Documentation": "https://github.com/docker/dredge-cli-repo/wiki",
        "Source Code": "https://github.com/docker/dredge-cli-repo",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "dredge": [
            "static/*.html",
            "static/*.css",
            "static/*.js",
        ]
    },
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": ["torch-cuda>=2.0.0"],
        "all": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "dredge=dredge.cli:main",
            "dredge-server=dredge.server:run",
            "dredge-advanced=dredge.server:run_advanced",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
    ],
    keywords=[
        "security",
        "vulnerability",
        "dependabot",
        "fibot",
        "dredge",
        "ai",
        "intelligence",
        "models",
        "quasimoto",
        "string-theory",
        "mcp",
    ],
    zip_safe=False,
)
