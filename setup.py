import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="alphazero-mtb2718",
    version="0.0.1",
    author="Matt Bauch",
    author_email="mtb2718@gmail.com",
    description="Alphazero implementation with extensions and additional experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mtb2718/alphazero",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)