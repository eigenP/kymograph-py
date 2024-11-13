from setuptools import setup, find_packages

setup(
    name='python_kymograph',  # Package name
    version='0.1.0',          # Version number
    description='A package for generating kymographs from image data.',
    author='Panagiotis Oikonomou',       # Your name
    author_email='po2236@columbia.edu',  # Your email
    packages=find_packages(), # Automatically find packages
    install_requires=[        # Dependencies
        'numpy',
        'pandas',
        'tqdm',
        'matplotlib',
        'scikit-image',
        'scipy',
    ],
    classifiers=[             # Optional metadata
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 3 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Python version requirement
)
