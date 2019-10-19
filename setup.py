from setuptools import setup

setup(name='kinematics_and_rotation',
      version='0.1rc0',
      description='Tools for kinematics',
      url='http://github.com/RuthAngus/kinematics-and-rotation',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['kinematics_and_rotation'],
      install_requires=['numpy', 'pandas', 'tqdm', 'astropy', 'matplotlib', 'scipy'],
      zip_safe=False)
