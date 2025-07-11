from setuptools import setup, find_packages
import os

# Read the README file for a long description.
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='python_kymograph',  # Package name
    version='0.1.0',          # Version number
    description='A package for generating kymographs from image data.',
    long_description=long_description,
    long_description_content_type="text/markdown",  # This tells PyPI the format of your long description
    author='Panagiotis Oikonomou',       # Your name
    author_email='po2236@columbia.edu',  # Your email
    packages=find_packages(), # Automatically find packages
    install_requires=[        # Dependencies
        'numpy',
        'tqdm',
        'scipy',
    ],
    classifiers=[             # Optional metadata
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 3 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Python version requirement
)
