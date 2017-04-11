'''
--------------------------------------------------------------------------------

    setup.py

--------------------------------------------------------------------------------
Copyright 2013-2017 Pierre Denis

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
       data_files = [(join(get_python_lib(),'lea'), [ 'COPYING', 'COPYING.LESSER' ] ) ],
       long_description = '''Lea is a Python package aiming at working with discrete probability distributions in an intuitive way. It allows you to model a broad range of random phenomenons, like dice throwing, coin tossing, gambling, weather, finance, etc. More generally, Lea may be used for any finite set of discrete values having known probability: numbers, booleans, date/times, symbols, ... Each probability distribution is modeled as a plain object, which can be named, displayed, queried or processed to produce new probability distributions.

Lea also provides advanced functions that target Probabilistic Programming (PP); these include conditional probabilities, Bayes inference and Markov chains. To ease interactive calculations, an optional PP language (PPL), called "Leapp", is included in the package; it extends Python syntax with few constructs to define and manipulate probabilistic models in an extremely concise way.

To install this beta version of Lea, type the following command:
::

  pip install lea==%s

Please go on project home page below for a comprehensive documentation.''' % VER
      )
