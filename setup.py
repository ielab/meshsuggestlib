from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='meshsuggestlib',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/ielab/meshsuggestlib',
    license='Apache 2.0',
    author='Shuai Wang, Hang Li',
    author_email='shuai.wang2@uq.edu.au',
    description='',
    python_requires='>=3.7',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "transformers>=4.10.0",
        "datasets>=1.1.3",
    ]
)
