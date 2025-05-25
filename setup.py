from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-rps-vision",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Intelligent Rock-Paper-Scissors detection with adaptive AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptive-rps-vision",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9", "mypy>=0.910"],
    },
    entry_points={
        "console_scripts": [
            "adaptive-rps-train=adaptive_rps.scripts.train:main",
            "adaptive-rps-app=adaptive_rps.scripts.run_app:main",
        ],
    },
)
