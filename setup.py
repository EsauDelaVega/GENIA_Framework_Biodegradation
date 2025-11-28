from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genia-framework",
    version="2.0.0",
    author="Esau De la Vega",
    author_email="your_email@tamu.edu",
    description="Machine learning framework for designing synthetic microbial communities for environmental bioremediation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EsauDelaVega/GENIA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6",
        "python-louvain>=0.15",
        "node2vec>=0.4.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "biopython>=1.79",
    ],
)
