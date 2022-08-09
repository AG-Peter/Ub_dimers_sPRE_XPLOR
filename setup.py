from setuptools import setup
import os
import re

# read _version.py to have a single source file for version.
exec(open('xplor/_version.py').read())

# setup
setup(name='xplor_functions',
      version=__version__,
      description="For functions (and classes) that might be useful when calling XPLOR's python interpreter from within normal python.",
      author='Kevin Sawade',
      url="https://github.com/kevinsawade/xplor_functions",
      packages=['xplor'],
      install_requires=[
          'numpy'
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
      ],
      scripts=['xplor/scripts/xplor_single_struct.py'],
      include_package_data=True,
      package_data = {
        'xplor': ['data/*']
      }
      )