"""SeismiPro is a library for seismic data processing."""

import re
from setuptools import setup, find_packages

with open('seismicpro/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='SeismicPro',
    packages=find_packages(exclude=['tutorials']),
    version=version,
    url='https://github.com/gazprom-neft/SeismicPro',
    license='Apache License 2.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='A framework for seismic data processing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    platforms='any',
    include_package_data=True,
    install_requires=[
        'numpy>=1.19.5',
        'scipy>=1.5.2',
        'numba>=0.53.1',
        'pandas>=1.1.5',
        'scikit-learn>=0.23.2',
        'opencv_python>=4.5.1',
        'networkx>=2.5',
        'segyio>=1.9.5',
        'tqdm>=4.56.0',
        'pytest>=6.0.1',
        'torch>=1.8',
        'matplotlib>=3.3.1',
        'seaborn>=0.11.1',
        'dill>=0.3.3',
        'multiprocess>=0.70.11',
        'requests>=2.24',
        'psutil>=5.7.2',
        'batchflow @ git+https://github.com/analysiscenter/batchflow.git@7853cfb#egg=batchflow',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
)
