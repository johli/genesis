import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genesis",
    version="0.1",
    author="Johannes Linder",
    author_email="johannes.linder@hotmail.com",
    description="Generative Functional Sequence Model",
    long_description=long_description,
    url="https://github.com/johli/genesis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
