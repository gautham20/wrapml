import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wrapml",
    version="0.0.1",
    author="gautham20",
    author_email="gautham.kumaran11@gmail.com",
    description="A wrapper for kaggle competitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gautham20/wrapml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)