'''
--------------------------------------------------------------------------------

    __init__.py

--------------------------------------------------------------------------------
Copyright 2013-2016 Pierre Denis

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

from .leaf import *
__all__ = ( # objects defined in lea.lea module
            'Lea', 'V', 'VP', 'B', 'P', 'Pf', 'X',
            # objects defined in lea.leaf module
            'die', 'dice', 'diceSeq', 'D6', 'flip', 'cardRank', 'cardSuite', 'card')
