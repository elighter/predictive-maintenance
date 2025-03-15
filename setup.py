from setuptools import setup, find_packages

setup(
    name="predictive_maintenance",
    version="0.1.0",
    description="Makine öğrenmesi ile endüstriyel bakım tahmini projesi",
    author="Prediktif Bakım Ekibi",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.3.0",
        "kagglehub>=0.2.0",
        "streamlit>=1.10.0",
        "shap>=0.40.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 