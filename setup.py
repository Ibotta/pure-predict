"""
Run setup
"""

from setuptools import setup, find_packages

DISTNAME = "pure-predict"
VERSION = "0.0.1"
DESCRIPTION = "Machine learning prediction in pure Python"
LONG_DESCRIPTION = ("""
pure-predict: Machine learning prediction in pure Python
========================================================

``pure-predict`` speeds up and slims down machine learning prediction applications. It is 
a foundational tool for serverless inference or small batch prediction with popular machine 
learning frameworks like `scikit-learn <https://scikit-learn.org/stable/>`__ and `fasttext <https://fasttext.cc/>`__. 
It implements the predict methods of these frameworks in pure Python.
""")
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering"
    ]
AUTHOR = "Ibotta Inc."
AUTHOR_EMAIL = "machine_learning@ibotta.com"
LICENSE = "Apache 2.0"
DOWNLOAD_URL = "https://pypi.org/project/pure-predict/#files"
PROJECT_URLS = {
    "Source Code": "https://github.com/Ibotta/pure-predict"
    }
MIN_PYTHON_VERSION = "3.5"

tests_require = [
    "xgboost>=0.82",
    "scikit-learn>=0.20",
    "pandas",
    "numpy",
    "fasttext",
    "pytest"
    ]

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      classifiers=CLASSIFIERS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      packages=find_packages(),
      python_requires=">={0}".format(MIN_PYTHON_VERSION),
      extras_require=dict(tests=tests_require)
      )
