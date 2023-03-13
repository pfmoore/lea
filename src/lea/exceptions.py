'''
--------------------------------------------------------------------------------

    exceptions.py

--------------------------------------------------------------------------------
Copyright 2013-2023 Pierre Denis

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

class LeaError(Exception):
    ''' exception representing any violation of requirements of Lea methods  
    '''
    pass
    
class _FailedRandomMC(Exception):
    ''' internal exception representing a failure to get a set of random values that
        satisfy a given condition in a given number of trials (see methods having '..._mc' suffix) 
    '''
    pass
