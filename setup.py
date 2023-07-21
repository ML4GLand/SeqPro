from setuptools import find_packages, setup

setup(name="seqpro", packages=find_packages())

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="seqpro",
    version="0.1.3",
    author="Adam Klie",
    author_email="aklie@.ucsd.edu",
    description="Sequence processing toolkit",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ML4GLand/SeqPro",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
