from setuptools import find_packages, setup

setup(
    name="knn-tspi",
    version="1.0.0",
    author="Gabriel Dalforno, Moisés Rocha",
    author_email="gdalforno7@gmail.com, moises0rocha@gmail.com",
    description="K-Nearest Neighbors Time Series Prediction with Invariances (KNN-TSPI)",  # noqa: E501
    long_description="Implementation of K-Nearest Neighbors Time Series Prediction with Invariances (KNN-TSPI) algorithm for univariate time series forecasting",  # noqa: E501
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.26.0"],
)
