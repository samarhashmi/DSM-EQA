from setuptools import setup, find_packages

setup(
    name="dsm_eqa",
    version="0.1.0",
    description="Data-Driven Structure Modeling with Equation Quality Assessment",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.10.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
    ],
)
