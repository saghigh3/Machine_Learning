""" MLROSe setup file."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause

from setuptools import setup

def readme():
    """
    Function to read the long description for the MLROSe package.
    """
    with open('README.md') as _file:
        return _file.read()

VERSION = '2.2.4'

setup(name='mlrose_hiive',
      version=VERSION,
      description="MLROSe: Machine Learning, Randomized Optimization and"
      + " Search (hiive extended remix)",
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/hiive/mlrose',
      author='Genevieve Hayes (modified by Andrew Rollings)',
      license='BSD',
      download_url='https://github.com/hiive/mlrose/archive/'+VERSION+'.tar.gz',
      classifiers=[
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English",
          "License :: OSI Approved :: BSD License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      packages=['mlrose_hiive','mlrose_hiive.runners','mlrose_hiive.generators', 'mlrose_hiive.algorithms',
                'mlrose_hiive.algorithms.decay', 'mlrose_hiive.algorithms.crossovers',
                'mlrose_hiive.opt_probs', 'mlrose_hiive.fitness', 'mlrose_hiive.algorithms.mutators',
                'mlrose_hiive.neural', 'mlrose_hiive.neural.activation', 'mlrose_hiive.neural.fitness',
                'mlrose_hiive.neural.utils', 'mlrose_hiive.decorators',
                'mlrose_hiive.gridsearch'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'networkx', 'joblib'],
      python_requires='>=3',
      zip_safe=False)
