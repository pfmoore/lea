'''
--------------------------------------------------------------------------------

    alea.py

--------------------------------------------------------------------------------
Copyright 2013 Pierre Denis

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
from random import randrange
from math import log

class Alea(Lea):
    
    '''
    Alea is a Lea subclass, which instance is defined by explicit probability distribution data.
    An Alea instance is defined by given value-probability pairs. Each probability is
    defined as a positive "counter" or "weight" integer, without upper limit. The actual
    probabilities are calculated by dividing the counters by the sum of all counters.
    '''

    __slots__ = ('_vps','_count')
    
    def __init__(self,vps):
        Lea.__init__(self)
        self._alea = self
        self._vps = tuple(vps)
        self._count = sum(p for (v,p) in self._vps)

    @staticmethod
    def fromValFreqsDict(probDict):
        count = sum(probDict.itervalues())
        if count == 0:
            raise Exception("ERROR: no value")
        gcd = count
        impossibleValues = []
        for (v,p) in probDict.iteritems():
            if p < 0:
                raise Exception("ERROR: negative probability")
            if p == 0:
                impossibleValues.append(v)
            elif gcd > 1:
                while p != 0:
                    if gcd > p:
                        (gcd,p) = (p,gcd)
                    p %= gcd
        for impossibleValue in impossibleValues:
            del probDict[impossibleValue]
        try:            
            return Alea(sorted((v,p/gcd) for (v,p) in probDict.iteritems()))
        except TypeError:
            # no ordering relationship on values (e.g. complex numbers)
            return Alea((v,p/gcd) for (v,p) in probDict.iteritems())
            
    @staticmethod
    def fromVals(*values):
        probDict = {}
        for value in values:
            probDict[value] = probDict.get(value,0) + 1
        return Alea.fromValFreqsDict(probDict)

    @staticmethod
    def fromValFreqs(*valueFreqs):
        probDict = {}
        for (value,freq) in valueFreqs:
            probDict[value] = probDict.get(value,0) + freq
        return Alea.fromValFreqsDict(probDict)

    def __str__(self):
        vm = max(len(str(v)) for (v,p) in self._vps)
        pm = len(str(max(p for (v,p) in self._vps)))
        if self._count == 1:
            den = ''
        else:
            den = '/%d' % self._count
        fmt = ("%%%ds : %%%dd" % (vm,pm)) + den
        return "\n".join(fmt%vp for vp in self._vps)

    def asPct(self,nbDecimals=1):
        vm = max(len(str(v)) for (v,p) in self._vps)
        fmt = "%%%ds : %%%d.%df %%%%" % (vm,4+nbDecimals,nbDecimals)
        count = float(self._count)
        return "\n".join(fmt%(v,100.*p/count) for (v,p) in self._vps)

    def _reset(self):
        pass

    def _clone(self,cloneTable):
        return Alea(self._vps)
        
    def _genVPs(self):
        for vp in self._vps:
            yield vp

    def _p(self,val):
        for (v,p) in self._vps:
            if v == val:
                return (p,self._count)
        return (0,self._count)

    def integral(self):
        ''' returns a tuple with couples (x,p) giving the probability weight p of having a value
            less than or equal to x;
            if an order relationship is defined on values, then the tuples follows the increasing
            order of x; otherwise, an arbitrary order is used
        '''
        res = []
        f = 0.0
        for (v,p) in self._vps:
            f += p
            res.append((v,f))
        return tuple(res)

    def _random(self,integral):
        r = randrange(self._count)
        f0 = 0
        for (x,f1) in integral:
            if f0 <= r < f1:
                return(x)
            f0 = f1

    def random(self,n=None):
        ''' if n is None, returns a random value with the probability given by the distribution
            otherwise returns a tuple of n such random values
        '''
        integral = self.integral()
        if n is None:
            return self._random(integral)
        return tuple(self._random(integral) for i in xrange(n))

    def randomDraw(self,n=None,sorted=False):
        ''' if n is None, returns a tuple with all the values of the distribution,
            in a random order respecting the probabilities
            (the higher count of a value, the most likely the value will be in the
             beginning of the sequence)
            if n > 0, then only n different values will be drawn
            if sorted is True, then the returned tuple is sorted
        '''
        if n is None:
           n = len(self._vps)
        lea = self
        res = []
        while n > 0:
            n -= 1
            lea = lea.getAlea()
            x = lea.random()
            res.append(x)
            lea = lea.given(lea!=x)
        if sorted:
            res.sort()
        return tuple(res)

    def mean(self):
        ''' returns the mean value of the probability distribution, which is the
            probability weighted sum of the values;
            requires that
            1 - the values can be subtracted together,
            2 - the differences of values can be multiplied by floating-point numbers,
            3 - the differences of values multiplied by floating-point numbers can be
                added to the values;
            if any of these conditions is not met, then the result depends of the
            value class implementation (likely, raised exception)
        '''
        res = None
        x0 = None
        for (x,p) in self._vps:
            if x0 is None:
                x0 = x
            elif res is None:
                res = p * (x-x0)
            else:
                res += p * (x-x0)
        if res is not None:
            try:
                x0 += res / float(self._count)
            except:
                x0 += res / self._count
        return x0
   
    def variance(self):
        ''' returns the variance of the probability distribution;
            requires that
            1 - the requirements of the mean() method are met,
            2 - the values can be subtracted to the mean value,
            3 - the differences between values and the mean value can be squared;
            if any of these conditions is not met, then the result depends of the
            value implementation (likely, raised exception)
        '''
        res = 0
        m = self.mean()
        for (v,p) in self._vps:
            res += p*(v-m)**2
        return res / float(self._count)    

    def entropy(self):
        ''' returns the entropy of the distribution
        '''
        res = 0
        count = float(self._count)
        for (v,p) in self._vps:
            if p > 0:
               p /= count
               res -= p*log(p)
        return res / log(2)  

