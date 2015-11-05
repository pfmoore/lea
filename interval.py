'''
--------------------------------------------------------------------------------

    interval.py

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

from lea import Lea
from alea import Alea
from toolbox import cmp
from math import isnan

inf = float('inf')
nan = float('nan')

class Interval(object):
    
    '''
    An Interval instance represents an interval of numbers between two given
    bounds. The bounds can be any integer or float numbers, such that
    lower bound <= higher bound. The bounds are inclusive except if they 
    are infinite, which are represented by -inf and + inf., i.e. infinite
    numbers do NOT belong to the interval that they are bounding.
    Nothing is known on the distribution of numbers in the interval nor
    whether specific numbers shall be excluded (e.g. non-integers).  
    Usual arithmetic operations (plus, minus, etc) can be used on intervals
    producing new intervals.
    '''
    
    class Error(Exception):
        ''' exception representing any violation of requirements of Interval methods  
        '''
        pass

    #_instancesDict = dict()    
    # no: shall take lea as argument, __init__, Ieterval build, clone
    # getLoVal: rteurn min val (cache)
    # getHiVal: rteurn max val (cache)
    
    '''
    Warning: the following shall be enforced:
    Interval.build(0,inf) * 0    --> 0              OK
    Interval.build(0,inf) / inf  --> i[0,inf)       NOK
    Interval.build(0,inf) /-inf  --> i(-inf,0]      NOK
    
    min(3,nan) = 3
    max(3,nan) = 3
    implement this to remove any occurrence of nan:
    0*inf  = 0
    0*-inf = 0
    +x/0 = +inf
    -x/0 = -inf
    +inf/+inf = +inf
    -inf/-inf = +inf
    +inf/-inf = -inf
    -inf/+inf = -inf    
    '''
    
    __slots__ = ('_lea1',)
    
    def __init__(self,lea1):
        ''' initializes a new interval instance, with given Lea instance
            having just loVal and hiVal as values; their respective
            probabilities are irrelevant
        '''
        # note that the choice of storing Lea instead of simple loVal, hiVal
        # allows to reuse the binding mechanism of Lea if the same Interval
        # instance appears several times in the same expression;
        # this means that, although the distribution of numbers in a given
        # interval i is unknown, the same number is assumed to be bound on i;
        # example: i == i             shall be true
        #          i - i == 0         shall be true        
        #      but i == i.clone()     shall be undefined (raised exception) 
        #          i - i.clone() == 0 shall be undefined (raised exception) 
        self._lea1 = lea1
    
    def clone(self):
        ''' returns a clone of the current interval, without any binding with it 
        '''
        return Interval(self._lea1.clone())

    @staticmethod
    def build(loVal,hiVal):
        ''' returns a new interval for the given bounds
        '''
        #if loVal == hiVal:
        #    return loVal
        if loVal > hiVal:
            raise Interval.Error('impossible to build i[%s,%s] interval'%(loVal,hiVal))
        return Interval(Alea(((loVal,1),(hiVal,1))))
    
    def getLoHiVals(self):
        ''' returns a tuple (loVal,hiVal) with lower and higher bounds of current interval
        '''
        vs = tuple(v for (v,p) in self._lea1.genVPs())
        if any(isnan(v) for v in vs):
            raise Interval.Error('impossible to calculate interval')
        return (min(vs),max(vs))

    def getLoHiVals2(self,other):
        ''' returns a tuple ((loVal1,hiVal1),(loVal1,hiVal1))
            with, as first element, lower and higher bounds of current interval
            as second element, lower and higher bounds of given other if interval
             or (other,other) if not interval
        '''
        if isinstance(other,Interval):
            loHiVals2 = other.getLoHiVals()
        else:
            loHiVals2 = (other,other)
        return (self.getLoHiVals(),loHiVals2)

    def __str__(self):
        ''' returns a string representation of the interval, using standard
            mathematical notation prefixed with 'i'
        '''
        (loVal,hiVal) = self.getLoHiVals()
        if loVal == hiVal:
            return str(loVal)
        if loVal == -inf:
            res = 'i(-inf'
        else:
            res = 'i['+str(loVal)
        res += ','
        if hiVal == inf:
            res += '+inf)'
        else:
            res += str(hiVal) + ']'
        return res
        
    __repr__ = __str__

    def intersects(self,other):
        ''' returns true iff current interval
            - intersects other if other is an interval
            - contains other if other is a number
        '''
        if self is other:
            return True
        ((loVal1,hiVal1),(loVal2,hiVal2)) = self.getLoHiVals2(other)
        return max(loVal1,loVal2) <= min(hiVal1,hiVal2)
        
    def merge(self,other):
        ''' 
        '''
        if self is other:
            return self
        ((loVal1,hiVal1),(loVal2,hiVal2)) = self.getLoHiVals2(other)
        return Interval.build(min(loVal1,loVal2),max(hiVal1,hiVal2))

    @staticmethod
    def getV(obj):
        ''' 
        '''
        if isinstance(obj,Interval):
            return obj._lea1
        return obj
    
    def __buildFunc1(f1):
        ''' 
        '''
        return lambda self: Interval(f1(self._lea1))

    def __buildFunc2(f2):
        ''' 
        '''
        return lambda self,other: Interval(f2(self._lea1,Interval.getV(other)))
    
    def __buildBoolFunc2(f2,opStr):
        ''' 
        '''
        def f2r(self,other):
            vs = frozenset(v for (v,p) in f2(self._lea1,Interval.getV(other)).genVPs())
            if len(vs) == 1:
                return True in vs
            raise Interval.Error("impossible to evaluate '%s %s %s'"%(self,opStr,other))
        return f2r

    __eq__ = __buildBoolFunc2(Lea.__eq__,'==')
    __ne__ = __buildBoolFunc2(Lea.__ne__,'!=')
    __lt__ = __buildBoolFunc2(Lea.__lt__,'<')
    __le__ = __buildBoolFunc2(Lea.__le__,'<=')
    __gt__ = __buildBoolFunc2(Lea.__gt__,'>')
    __ge__ = __buildBoolFunc2(Lea.__ge__,'>=')
    __pos__ = __buildFunc1(Lea.__pos__)
    __neg__ = __buildFunc1(Lea.__neg__)
    # __abs__ = __buildFunc1(Lea.__abs__)
    __add__ = __buildFunc2(Lea.__add__)
    __radd__ = __buildFunc2(Lea.__radd__)
    __sub__ = __buildFunc2(Lea.__sub__)
    __rsub__ = __buildFunc2(Lea.__rsub__)
    __mul__ = __buildFunc2(Lea.__mul__)
    __rmul__ = __buildFunc2(Lea.__rmul__)
    __pow__ = __buildFunc2(Lea.__pow__)
    __rpow__ = __buildFunc2(Lea.__rpow__)
    __truediv__ = __buildFunc2(Lea.__truediv__)
    __rtruediv__ = __buildFunc2(Lea.__rtruediv__)
    __floordiv__ = __buildFunc2(Lea.__floordiv__)
    __rfloordiv__ = __buildFunc2(Lea.__rfloordiv__)
    
    # ?????????
    def __mod__(self,other):
        ''' 
        '''
        raise Interval.Error("% operator not supported on intervals")

    def __rmod__(self,other):
        ''' 
        '''
        raise Interval.Error("% operator not supported on intervals")     

    def __divmod__(self,other):
        ''' 
        '''
        raise Interval.Error("divmod function not supported on intervals")

    def __rdivmod__(self,other):
        ''' 
        '''
        raise Interval.Error("divmod function not supported on intervals")

    def __abs__(self):
        ''' 
        '''
        res = Interval(Lea.__abs__(self._lea1))
        (loVal,hiVal) = self.getLoHiVals()
        if hiVal <= 0:
            return Interval.build(-hiVal,-loVal)
        if 0 <= loVal:
            return Interval.build(loVal,hiVal)
        return Interval.build(0,max(-loVal,hiVal))

    # Python 2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    
    def __hash__(self):
        ''' 
        '''
        return id(self)
    
    @staticmethod
    def cmpForSort(i1,i2):
        ''' 
        '''
        if not isinstance(i1,Interval):
            if not isinstance(i2,Interval):
                return cmp(i1,i2)
            return -Interval.cmpForSort(i2,i1)
        ((loVal1,hiVal1),(loVal2,hiVal2)) = i1.getLoHiVals2(i2)
        if (loVal1 == loVal2) and (hiVal1 == hiVal2):
            return 0
        if (loVal1 < loVal2) or (loVal1 == loVal2 and loVal2 == hiVal2):
           return -1
        return +1
            
    '''
    def __new__(cls,loVal,hiVal):
        if loVal > hiVal:
            raise Interval.Error('impossible to build [%s,%s] interval'%(loVal,hiVal))
        if loVal == hiVal:
            return loVal
        interval = object.__new__(cls)
        interval.loVal = loVal
        interval.hiVal = hiVal
        interval._alea = Lea.fromVals(loVal,hiVal) 
        return interval
        
    def __str__(self):
        if self.loVal == -inf:
            res = '(-inf'
        else:
            res = '['+str(self.loVal)
        res += ','
        if self.hiVal == inf:
            res += '+inf)'
        else:
            res += str(self.hiVal) + ']'
        return res
        
    __repr__ = __str__
    
    def intersects(self,other):
        if self is other:
            return True
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        return max(self.loVal,otherLoVal) <= min(self.hiVal,otherHiVal)
        
    def merge(self,other):
        if self is other:
            return self
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        return Interval(min(self.loVal,otherLoVal),max(self.hiVal,otherHiVal))
    
    @staticmethod
    def getLoHiVals(obj):
        if isinstance(obj,Interval):
            return (obj.loVal,obj.hiVal)
        return (obj,obj)        
    
    @staticmethod
    def getV(obj):
        if isinstance(obj,Interval):
            return obj._alea
        return obj        
    
    @staticmethod
    def buildFromLea(lea1):
        vs = tuple(v for (v,p) in lea1.genVPs())
        return Interval(min(vs),max(vs))     
    
    def __build(f):
        return lambda self,other: Interval.buildFromLea(f(self._alea,Interval.getV(other)))
    
    __add__ = __build(Lea.__add__)
    __radd__ = __build(Lea.__radd__)
    __sub__ = __build(Lea.__sub__)
    __rsub__ = __build(Lea.__rsub__)
    
    def __add__(self,other):
        return Interval.buildFromLea(Lea.__add__(self._alea,Interval.getV(other)))

    def __radd__(self,other):
        return Interval.buildFromLea(Lea.__radd__(self._alea,Interval.getV(other)))
          
    def __sub__(self,other):
        return Interval.buildFromLea(Lea.__sub__(self._alea,Interval.getV(other)))

    def __rsub__(self,other):
        return Interval.buildFromLea(Lea.__rsub__(self._alea,Interval.getV(other)))
    '''

    '''
    
    def __mul__(self,other):
        return Interval.buildFromLea(self._alea*other._alea)

    def __rmul__(self,other):
        return Interval.buildFromLea(other._alea*self._alea)

    def __truediv__(self,other):
        return Interval.buildFromLea(self._alea/other._alea)

    def __rtruediv__(self,other):
        return Interval.buildFromLea(other._alea/self._alea)

    def __floordiv__(self,other):
        return Interval.buildFromLea(self._alea//other._alea)

    def __rfloordiv__(self,other):
        return Interval.buildFromLea(other._alea//self._alea)

    # Python 2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    
    def __pos__(self):
        return Interval.buildFromLea(+self._alea)

    def __neg__(self):
        return Interval.buildFromLea(-self._alea)

    def __eq__(self,other):
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        if (self.loVal == otherLoVal) and (self.hiVal == otherHiVal):
            return True
        if max(self.loVal,otherLoVal) > min(self.hiVal,otherHiVal):
            return False
        raise Interval.Error("%s == %s"%(self,other))

    def __ne__(self,other):
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        if (self.loVal == otherLoVal) and (self.hiVal == otherHiVal):
            return False
        if max(self.loVal,otherLoVal) > min(self.hiVal,otherHiVal):
            return True
        raise Interval.Error("%s != %s"%(self,other))

    def __lt__(self,other):
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        if self.hiVal < otherLoVal:
            return True
        if otherHiVal <= self.loVal:
            return False
        raise Interval.Error("%s < %s"%(self,other))    

    def __le__(self,other):
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        if self.hiVal <= otherLoVal:
            return True
        if otherHiVal < self.loVal:
            return False
        raise Interval.Error("%s <= %s"%(self,other))    

    def __gt__(self,other):
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        if self.loVal > otherHiVal:
            return True
        if otherLoVal >= self.hiVal:
            return False
        raise Interval.Error("%s > %s"%(self,other))

    def __ge__(self,other):
        (otherLoVal,otherHiVal) = Interval.getLoHiVals(other)
        if self.loVal >= otherHiVal:
            return True
        if otherLoVal > self.hiVal:
            return False
        raise Interval.Error("%s >= %s"%(self,other))

    @staticmethod
    def cmpForSort(i1,i2):
        if not isinstance(i1,Interval):
            if not isinstance(i2,Interval):
                return cmp(i1,i2)
            return -Interval.cmpForSort(i2,i1)
        (i2LoVal,i2HiVal) = Interval.getLoHiVals(i2)
        if (i1.loVal == i2LoVal) and (i1.hiVal == i2HiVal):
            return 0
        if (i1.loVal < i2LoVal) or (i1.loVal == i2LoVal and i2LoVal == i2HiVal):
           return -1
        return +1
        
    def __hash__(self):
        return id(self)
    '''        ''' 
        '''
