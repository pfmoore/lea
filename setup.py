'''
--------------------------------------------------------------------------------

    setup.py

--------------------------------------------------------------------------------
Copyright 2013-2019 Pierre Denis

This file is part of Lea.

Lea is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Lea is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Lea.  If not, see <http://www.gnu.org/licenses/>.
--------------------------------------------------------------------------------
'''

from distutils.core import setup
from distutils.sysconfig import get_python_lib
from os.path import join

from license import VER

setup( name = 'lea',
       version = VER,
       description = 'Discrete probability distributions in Python',
       author = 'Pierre Denis',
       author_email = 'pie.denis@skynet.be',
       url = 'http://bitbucket.org/piedenis/lea',
       license = 'LGPL',
       keywords = ['probability', 'discrete', 'distribution', 'probabilistic programming'],
       packages = [ 'lea' ],
       package_dir = {'lea': ''},
       data_files = [(join(get_python_lib(),'lea'), [ 'COPYING', 'COPYING.LESSER', 'README.md' ] ) ],
       classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Topic :: Utilities",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Education",
          "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)"
       ],
       long_description = '''Lea is a Python module aiming at working with discrete probability distributions in an intuitive way.

It allows you modeling a broad range of random phenomena: gambling, weather, finance, etc. More generally, Lea may be used for any finite set of discrete values having known probability: numbers, booleans, date/times, symbols, ... Each probability distribution is modeled as a plain object, which can be named, displayed, queried or processed to produce new probability distributions.

Lea also provides advanced functions and Probabilistic Programming (PP) features; these include conditional probabilities, JPD, CPT, BN, Markov chains and symbolic computation.

Lea can be used for AI, machine learning, education, ...

To install Lea {0}, type the following command:
::

  pip install lea=={0}

Please go on Lea project page (beside) for a comprehensive documentation.'''.format(VER)
      )
