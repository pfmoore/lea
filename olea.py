from itertools import izip
from flea import Flea

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
        for (attrName,arg) in izip(aClass.__slots__,args):
            setattr(self,attrName,arg)
            length = len(str(arg))
            maxLengthAttrName = '__maxLength' + attrName
            if length > getattr(aClass,maxLengthAttrName):
                setattr(aClass,maxLengthAttrName,length)
                aClass.__templateStr = "<%s>" % ', '.join("%s=%%%ds"%(attrName,getattr(aClass,'__maxLength'+attrName)) for attrName in aClass.__slots__)

    def __str__(self):
        aClass = self.__class__
        return aClass.__templateStr % tuple(getattr(self,attrName) for attrName in aClass.__slots__)
    __repr__ = __str__

    def __hash__(self):
        return hash(tuple(getattr(self,attrName) for attrName in self.__class__.__slots__))
    
    def __cmp__(self,other):
        for attrName in self.__class__.__slots__:
            res = cmp(getattr(self,attrName),getattr(other,attrName))
            if res != 0:
                break
        return res
        
