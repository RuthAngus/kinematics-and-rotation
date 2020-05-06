from setuptools import setup

setup(name='cds',
      version='0.1rc0',
      description='Make machine readable table for Visier',
      packages=['cds'],
      install_requires=['numpy', "astropy"],
      zip_safe=False)
