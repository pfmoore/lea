'''
--------------------------------------------------------------------------------

    alea.py

--------------------------------------------------------------------------------
Copyright 2013, 2014 Pierre Denis

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
from prob_fraction import ProbFraction
from random import randrange
from math import log, sqrt, exp
from collections import defaultdict
from toolbox import LOG2
import operator

class Alea(Lea):
    
    '''
    Alea is a Lea subclass, which instance is defined by explicit probability distribution data.
    An Alea instance is defined by given value-probability pairs. Each probability is
    defined as a positive "counter" or "weight" integer, without upper limit. The actual
    probabilities are calculated by dividing the counters by the sum of all counters.
    '''

    __slots__ = ('_vps','_count','_cumul','_invCumul','_optCumul')
    
    def __init__(self,vps):
        ''' initializes Alea instance's attributes
        '''
        Lea.__init__(self)
        self._alea = self
        self._vps = tuple(vps)
        self._count = sum(p for (v,p) in self._vps)
        self._cumul = None
        self._invCumul = None
        self._optCumul = None

    # constructor methods
    # -------------------
    
    def getAleaClone(self):
        ''' same as getAlea method, excepting if applied on an Alea instance:
            in this case, a clone of the Alea instance is returned (insead of itself)
        '''        
        return self.clone()

    @staticmethod
    def fromValFreqsDictGen(probDict):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob}, where prob is an integer number,
            a floating-point number or a fraction (Fraction or ProbFraction instance)
            so that each value val has probability proportional to prob to occur
            any value with null probability is ignored (hence not stored)
            if the sequence is empty, then an exception is raised
        '''
        probFractions = tuple(ProbFraction.coerce(probWeight) for probWeight in probDict.values())
        # TODO Check positive
        probWeights = ProbFraction.getProbWeights(probFractions)
        return Alea.fromValFreqsDict(dict(zip(probDict.keys(),probWeights)))
    
    @staticmethod
    def fromValFreqsDict(probDict,reducing=True):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob}, where prob is an integer number
            so that each value val has probability proportional to prob to occur
            any value with null probability is ignored (hence not stored)
            if reducing is True, then the probability weights are reduced to have a GCD = 1
            if the sequence is empty, then an exception is raised
        '''
        count = sum(probDict.values())
        if count == 0:
            raise Lea.Error("impossible to build a probability distribution with no value")
        gcd = count
        impossibleValues = []
        # check probabilities, remove null probabilities and calculate GCD
        for (v,p) in probDict.items():
            if p < 0:
                raise Lea.Error("negative probability")
            if p == 0:
                impossibleValues.append(v)
            elif reducing and gcd > 1:
                while p != 0:
                    if gcd > p:
                        (gcd,p) = (p,gcd)
                    p %= gcd
        for impossibleValue in impossibleValues:
            del probDict[impossibleValue]
        vpsIter = probDict.items()
        if reducing:
            vpsIter = ((v,p//gcd) for (v,p) in vpsIter)
        vps = list(vpsIter)
        try:            
            vps.sort()
        except:
            # no ordering relationship on values (e.g. complex numbers)
            pass
        return Alea(vps)
            
    @staticmethod
    def fromVals(*values):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of values, so that each value occurrence is
            taken as equiprobable;
            if each value occurs exactly once, then the distribution is uniform,
            i.e. the probability of each value is equal to 1 / #values;
            if the sequence is empty, then an exception is raised
        '''
        probDict = {}
        for value in values:
            probDict[value] = probDict.get(value,0) + 1
        return Alea.fromValFreqsDict(probDict)

    @staticmethod
    def _fromValFreqs(valueFreqs,reducing):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            if reducing is True, then the frequencies are reduced by dividing them by
            their GCD
            if the sequence is empty, then an exception is raised
        '''        
        probDict = {}
        for (value,freq) in valueFreqs:
            probDict[value] = probDict.get(value,0) + freq
        return Alea.fromValFreqsDict(probDict,reducing)

    @staticmethod
    def fromValFreqs(*valueFreqs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            the frequencies are reduced by dividing them by their GCD
            if the sequence is empty, then an exception is raised
        '''        
        return Alea._fromValFreqs(valueFreqs,True)

    @staticmethod
    def fromValFreqsNR(*valueFreqs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            if the sequence is empty, then an exception is raised
        '''
        return Alea._fromValFreqs(valueFreqs,False)

    @staticmethod
    def poisson(mean):
        ''' static method, returns an Alea instance representing a Poisson probability
            distribution having the given mean; the distribution is approximated by
            the finite set of values that have non-null probability float representation
            (i.e. high values with too small probabilities are dropped)
        '''
        # TODO: improve implementation
        from sys import maxsize
        valFreqs = []
        p = exp(-mean)
        v = 0
        t = 0.
        while True:
            n = int(p*maxsize)
            if n <= 0:
                break
            valFreqs.append((v,n))
            t += p
            v += 1
            p = (p*mean) / v
        return Alea.fromValFreqs(*valFreqs)

    def asString(self,kind='/',nbDecimals=6,histoSize=100):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability in a format depending of given kind, which is string among
            '/', '.', '%', '-', '/-', '.-', '%-'; 
            the probabilities are displayed as
            - if kind[0] is '/' : rational numbers "n/d" or "0" or "1"
            - if kind[0] is '.' : decimals with given nbDecimals digits
            - if kind[0] is '%' : percentage decimals with given nbDecimals digits
            - if kind[0] is '-' : histogram bar made up of repeated '-', such that
                                  a bar length of histoSize represents a probability 1 
            if kind[1] is '-', the histogram bars with '-' are appended after 
                               numerical representation of probabilities
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used
        '''
        if kind not in ('/', '.', '%', '-', '/-', '.-', '%-'):
            raise Lea.Error("invalid display format '%s'"%kind)
        valueStrings = tuple(str(v) for (v,p) in self._vps)
        ps = tuple(p for (v,p) in self._vps)
        vm = max(len(v) for v in valueStrings)
        linesIter = (v.rjust(vm)+' : ' for v in valueStrings)
        probRepresentation = kind[0]
        withHisto = kind[-1] == '-'
        if probRepresentation == '/':
            pStrings = tuple(str(p) for p in ps)
            pm = len(str(max(p for p in ps)))
            if self._count == 1:
                den = ''
            else:
                den = '/%d' % self._count
            linesIter = (line+pString.rjust(pm)+den for (line,pString) in zip(linesIter,pStrings))
        else:
            c = float(self._count)    
            if probRepresentation == '.':
                fmt = "%%s%%.%df" % nbDecimals
                linesIter = (fmt%(line,p/c) for (line,p) in zip(linesIter,ps))
            elif probRepresentation == '%':
                fmt = "%%s%%%d.%df %%%%" % (4+nbDecimals,nbDecimals)
                linesIter = (fmt%(line,100.*p/c) for (line,p) in zip(linesIter,ps))    
        if withHisto:
            c = float(self._count)    
            linesIter = (line+' '+int(0.5+(p/c)*histoSize)*'-' for (line,p) in zip(linesIter,ps))
        return '\n'.join(linesIter)

    def __str__(self):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value  with its
            probability expressed as a rational number "n/d" or "0" or "1";
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
            called on evalution of "str(self)" and "repr(self)"
        '''
        return self.asString()
          
    def asFloat(self,nbDecimals=6):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as decimal with given nbDecimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.asString('.',nbDecimals)
        
    def asPct(self,nbDecimals=1):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as percentage with given nbDecimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.asString('%',nbDecimals)

    def histo(self,size=100):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as a histogram bar made up of repeated '-',
            such that a bar length of given size represents a probability 1
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.asString('-',histoSize=size)
        
    def _reset(self):
        pass

    def _clone(self,cloneTable):
        return Alea(self._vps)
        
    def _genVPs(self):
        return iter(self._vps)

    def _p(self,val):
        for (v,p) in self._vps:
            if v == val:
                return (p,self._count)
        return (0,self._count)

    def cumul(self):
        ''' returns a tuple with couples (x,p) giving the probability weight p that self <= x ;
            the sequence follows the order defined on values (if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        if self._cumul is None:
            cumulList = []
            pSum = 0
            for (v,p) in self._vps:
                pSum += p
                cumulList.append((v,pSum))
            self._cumul = tuple(cumulList)
        return self._cumul
        
    def invCumul(self):
        ''' returns a tuple with couples (x,p) giving the probability weight p that self >= x ;
            the sequence follows the order defined on values (if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        if self._invCumul is None:
            invCumulList = []
            pSum = self._count
            for (v,p) in self._vps:
                invCumulList.append((v,pSum))
                pSum -= p
            self._invCumul = tuple(invCumulList)
        return self._invCumul
        
    def optCumul(self):
        ''' returns a tuple with couples (x,p) giving the probability weight p that self equals x 
            or precedes it in the sequence of values with decreasing probabilities (i.e. the first
            couples (x,p) are those having the most likely x)
            Note : this method is used vs cumul method to speed up generation of random values
            Note : the returned value is cached
        '''        
        if self._optCumul is None:
            optCumulList = []
            pSum = 0
            for (v,p) in sorted(self._vps,key=operator.itemgetter(1),reverse=True):
                pSum += p
                optCumulList.append((v,pSum))
            self._optCumul = tuple(optCumulList)
        return self._optCumul

    def _randomVal(self,cumul):
        ''' returns a random value among the values of self, according to the probabilities
            given in given cumul (see optCumul method)
        '''
        r = randrange(self._count)
        f0 = 0
        for (x,f1) in cumul:
            if f0 <= r < f1:
                return x
            f0 = f1

    def randomVal(self):
        ''' returns a random value among the values of self, according to their probabilities
        '''
        return self._randomVal(self.optCumul())

    def randomIter(self):
        ''' generates an infinite sequence of random values among the values of self,
            according to their probabilities
        '''
        optCumul = self.optCumul()
        while True:
            yield self._randomVal(optCumul)

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

    def pCumul(self,val):
        ''' returns, as an integer, the probability weight that self <= val
            note that it is not required that val is in the support of self
        '''
        cp = 0
        for (v,p) in self.cumul():
            if val < v:
                break
            cp = p
        return cp

    def pInvCumul(self,val):
        ''' returns, as an integer, the probability weight that self >= val
            note that it is not required that val is in the support of self
        '''
        for (v,p) in self.invCumul():
            if val <= v:
                return p
        return 0

    @staticmethod
    def fastExtremum(cumulFunc,*aleaArgs):
        ''' returns a new Alea instance giving the probabilities to have the extremum
            value (min or max) of each combination of the given Alea args;
            cumulFunc is the cumul function that determines whether max or min is
            used : respectively, Alea.pCumul or Alea.pInvCumul;
            the method uses an efficient algorithm (linear complexity), which is
            due to Nicky van Foreest; for explanations, see
            http://nicky.vanforeest.com/scheduling/cpm/stochasticMakespan.html
        '''
        if len(aleaArgs) == 1:
            return aleaArgs[0]
        if len(aleaArgs) == 2:
            (aleaArg1,aleaArg2) = aleaArgs
            valFreqsDict = defaultdict(int)
            for (v,p) in aleaArg1.genVPs():
                valFreqsDict[v] = p * cumulFunc(aleaArg2,v)
            for (v,p) in aleaArg2.genVPs():
                valFreqsDict[v] += (cumulFunc(aleaArg1,v)-aleaArg1._p(v)[0]) * p
            return Lea.fromValFreqsDict(valFreqsDict)
        return Alea.fastExtremum(cumulFunc,aleaArgs[0],Alea.fastExtremum(cumulFunc,*aleaArgs[1:]))

    # WARNING: the following methods are called without parentheses (see Lea.__getattr__)

    indicatorMethodNames = ('mean','var','std','entropy','information')

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
            WARNING: this method is called without parentheses
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
            x0 += res / float(self._count)
        return x0
   
    def var(self):
        ''' returns a float number representing the variance of the probability distribution;
            requires that
            1 - the requirements of the mean() method are met,
            2 - the values can be subtracted to the mean value,
            3 - the differences between values and the mean value can be squared;
            if any of these conditions is not met, then the result depends of the
            value implementation (likely, raised exception)
            WARNING: this method is called without parentheses
        '''
        res = 0
        m = self.mean
        for (v,p) in self._vps:
            res += p*(v-m)**2
        return res / float(self._count)    

    def std(self):
        ''' returns a float number representing the standard deviation of the distribution
            requires that the requirements of the variance() method are met
            WARNING: this method is called without parentheses
        '''      
        return sqrt(self.var)
        
    def entropy(self):
        ''' returns a float number representing the entropy of the distribution
            WARNING: this method is called without parentheses
        '''
        res = 0
        count = float(self._count)
        for (v,p) in self._vps:
            if p > 0:
               p /= count
               res -= p*log(p)
        return res / LOG2
