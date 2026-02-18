"""Setup configuration for Lingulu ML package."""
from setuptools import setup, find_packages

setup(
    name="lingulu-ml",
    version="1.0.0",
    description="Lingulu Machine Learning API for pronunciation assessment",
    author="Lingulu Team",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    python_requires=">=3.11",
    install_requires=[
        "Flask>=3.1.2",
        "torch>=2.10.0",
        "transformers>=5.1.0",
        "librosa>=0.11.0",
        "soundfile>=0.13.1",
        "numpy>=2.4.2",
        "boto3>=1.36.19",
        "gunicorn>=23.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-flask>=1.3.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "isort>=5.13.2",
            "mypy>=1.8.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
)
