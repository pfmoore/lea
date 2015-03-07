'''
--------------------------------------------------------------------------------

    olea.py

--------------------------------------------------------------------------------
Copyright 2013, 2014, 2015 Pierre Denis

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

from flea import Flea
from toolbox import zip, dict


class Olea(Flea):
    
    '''
    Olea is a Flea subclass, which instance represents a joint probability distribution.
    An Olea instance is built from a sequence of n given attribute names and a given Lea
    instance with n-tuples as values; the Olea instance carries an inner "data" class built
    from the given attribute names and each value of the Olea instance is an instance
    of this class.
    '''

    __slots__ = ('_lea1','_attrNames','_class')

    def __init__(self,attrNames,lea1):
        ''' each value of lea1 is a tuple having same cardinality
            as attrNames
        '''
        self._attrNames = attrNames
        self._buildClass()
        Flea.__init__(self,self._class,lea1)

    def _clone(self,cloneTable):
        return Olea(self._attrNames,self._cleaArgs.clone(cloneTable))

    def _buildClass(self):
        classAttrDict = dict(('__maxLength'+attrName,0) for attrName in self._attrNames)
        classAttrDict['__slots__'] = tuple(self._attrNames)
        self._class = type('',(_TemplateClass,),classAttrDict)


class _TemplateClass(object):
    
    def __init__(self,*args):
        object.__init__(self)
        aClass = self.__class__
        for (attrName,arg) in zip(aClass.__slots__,args):
            setattr(self,attrName,arg)
            length = len(str(arg))
            maxLengthAttrName = '__maxLength' + attrName
            if length > getattr(aClass,maxLengthAttrName):
                setattr(aClass,maxLengthAttrName,length)
                aClass.__templateStr = "<%s>" % ', '.join("%s=%%%ds"%(attrName,getattr(aClass,'__maxLength'+attrName)) for attrName in aClass.__slots__)

    def __str__(self):
        return self.__class__.__templateStr % self._cmpkey()
    __repr__ = __str__
    
    def _cmpkey(self):
        return tuple(getattr(self,attrName) for attrName in self.__class__.__slots__)

    def __hash__(self):
        return hash(self._cmpkey())

    def __eq__(self, other):
        return self._cmpkey() == other._cmpkey()

    def __lt__(self, other):
        return self._cmpkey() < other._cmpkey()
    
    # Python 2 compatibility    
    def __cmp__(self,other):
        for attrName in self.__class__.__slots__:
            v1 = getattr(self,attrName)
            v2 = getattr(other,attrName)
            res = (v1 > v2) - (v1 < v2)
            if res != 0:
                break
        return res
