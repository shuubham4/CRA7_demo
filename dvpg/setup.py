from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['dvpg'],
    package_dir={'': 'website'})

setup(**setup_args)