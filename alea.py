'''
--------------------------------------------------------------------------------

    alea.py

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

from .lea import Lea
from .flea2 import Flea2
from .prob_fraction import ProbFraction
from random import randrange
from bisect import bisect_left, bisect_right
from itertools import combinations, combinations_with_replacement
from math import log, sqrt, exp, factorial
from .toolbox import LOG2, memoize, zip, next, dict, defaultdict, calcLCM, makeTuple
import operator

# try to import matplotlib package, required by plot() method
# if missing, no error is reported until plot is called
try:
    import matplotlib.pyplot as plt
    # switch on interactive mode, so the control is back to console as soon as a chart is displayed
    plt.ion()
except:
    pass

class Alea(Lea):
    
    '''
    Alea is a Lea subclass, which instance is defined by explicit probability distribution data.
    An Alea instance is defined by given value-probability pairs. Each probability is
    defined as a positive "counter" or "weight" integer, without upper limit. The actual
    probabilities are calculated by dividing the counters by the sum of all counters.
    Values having null probability counters are dropped.
    '''

    __slots__ = ('_vs','_ps','_count','_cumul','_invCumul','_randomIter','_cachesByFunc')
    
    def __init__(self,vs,ps):
        ''' initializes Alea instance's attributes
            vs is a sequence of values
            ps is a sequence of probability weights (same size and order as ps)
        '''
        Lea.__init__(self)
        # for an Alea instance, the alea cache is itself
        self._alea = self
        self._vs = vs
        self._ps = ps
        self._count = sum(ps)
        self._cumul = [0]
        self._invCumul = []
        self._randomIter = self._createRandomIter()
        self._cachesByFunc = dict()

    # constructor methods
    # -------------------
    
    def new(self):
        ''' returns a new Alea instance which is an independent clone of self;
            note that the present method overloads Lea.new to be more efficient
        '''
        # note that the new Alea instance shares the immutable _vs and _ps attributes of self
        newAlea = Alea(self._vs,self._ps)
        # it can share also the mutable _cumul and _invCumul attributes of self (lists)
        newAlea._cumul = self._cumul
        newAlea._invCumul = self._invCumul
        return newAlea

    @staticmethod
    def fromValFreqsDictGen(probDict):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob}, where prob is an integer number,
            a floating-point number or a fraction (Fraction or ProbFraction instance)
            so that each value val has probability proportional to prob to occur
            any value with null probability is ignored (hence not stored)
            the values are sorted if possible (i.e. no exception on sort), 
            otherwise, the order of values is unspecified; 
            if the sequence is empty, then an exception is raised
        '''
        probFractions = tuple(ProbFraction.coerce(probWeight) for probWeight in probDict.values())
        # TODO Check positive
        probWeights = ProbFraction.getProbWeights(probFractions)
        return Alea.fromValFreqsDict(dict(zip(probDict.keys(),probWeights)))
    
    @staticmethod
    def _getVPsIter(vps,reducing):
        gcd = sum(p for (v,p) in vps)
        if gcd == 0:
            raise Lea.Error("impossible to build a probability distribution with no value")
        if not reducing:
            gcd = 1
        for (v,p) in vps:
            if p < 0:
                raise Lea.Error("negative probability")
            if gcd > 1 and p > 0:
                while p != 0:
                    if gcd > p:
                        (gcd,p) = (p,gcd)
                    p %= gcd
        return ((v,p//gcd) for (v,p) in vps if p > 0)        
    
    @staticmethod
    def fromValFreqsDict(probDict,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob}, where prob is an integer number
            so that each value val has probability proportional to prob to occur;
            any value with null probability is ignored (hence not stored)
            if the sequence is empty, then an exception is raised;
            the method admits 2 optional boolean arguments (kwargs), viz.
              sorting and reducing:
            * sorting (default:True): if True, then the values for displaying
            the distribution or getting the values will be sorted if possible
            (i.e. no exception on sort); otherwise, or if sorting=False, then
            the order of values is unspecified; 
            * reducing (default:True): if True, then the given frequencies are
            reduced by dividing them by their GCD, otherwise, they are kept
            unaltered;  
        '''
        sorting = kwargs.get('sorting',True)
        reducing = kwargs.get('reducing',True)
        # check probabilities, remove null probabilities and calculate GCD
        vpsIter = Alea._getVPsIter(tuple(probDict.items()),reducing)
        if sorting:
            vps = list(vpsIter)
            try:            
                vps.sort()
            except:
                # no ordering relationship on values (e.g. complex numbers)
                pass
        else:
            vps = vpsIter
        return Alea(*zip(*vps))

    @staticmethod
    def fromVals(*values,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of values, so that each value occurrence is
            taken as equiprobable;
            if each value occurs exactly once, then the distribution is uniform,
            i.e. the probability of each value is equal to 1 / #values;
            if the sequence is empty, then an exception is raised
            for treatment of optional kwargs keywords arguments, see doc of
            Alea.fromValFreqsDict;
        '''
        probDict = dict()
        for value in values:
            probDict[value] = probDict.get(value,0) + 1
        return Alea.fromValFreqsDict(probDict,**kwargs)

    @staticmethod
    def fromValFreqs(*valueFreqs,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            if the sequence is empty, then an exception is raised;
            for treatment of optional kwargs keywords arguments, see doc of
            Alea.fromValFreqsDict;
        '''        
        probDict = dict()
        for (value,freq) in valueFreqs:
            probDict[value] = probDict.get(value,0) + freq
        return Alea.fromValFreqsDict(probDict,**kwargs)
            
    @staticmethod
    def fromValFreqsOrdered(*valueFreqs,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural
            number so that each value is taken with the given frequency
            the frequencies are reduced by dividing them by their GCD;
            the values will be stored and displayed in the given order (no sort);
            if the sequence is empty, then an exception is raised;
            requires that each value has a unique occurrence;
            the method admits 2 optional boolean arguments (kwargs), viz.
              reducing and check:
            * reducing (default: True): if True, then the given frequencies are
            reduced by dividing them by their GCD, otherwise, they are kept
            unaltered;
            * check (default: True): if True and if a value occurs multiple
            times, then an exception is raised;        
        '''
        reducing = kwargs.get('reducing',True)
        check = kwargs.get('check',True)
        (vs,ps) = zip(*Alea._getVPsIter(valueFreqs,reducing))
        # check duplicates
        if check and len(frozenset(vs)) < len(vs):
            raise Lea.Error("duplicate values")
        return Alea(vs,ps)

    def times(self,n,op=operator.add):
        ''' returns a new Alea instance representing the current distribution
            operated n times with itself, through the given binary operator op;
            if n = 1, then a copy of self is returned;
            requires that n is strictly positive; otherwise, an exception is
            raised;
            note that the implementation uses a fast dichotomic algorithm,
            instead of a naive approach that scales up badly as n grows
        '''
        if n <= 0:
            raise Lea.Error("times method requires a strictly positive integer")
        if n == 1:
            return self.new()
        (n2,r) = divmod(n,2)
        alea2 = self.times(n2,op)
        resFlea2 = Flea2(op,alea2,alea2.new())
        if r == 1:
            resFlea2 = Flea2(op,resFlea2,self)
        return resFlea2.getAlea()

    def isUniform(self):
        ''' returns True  if the probability distribution is uniform,
                    False otherwise
        '''
        p0 = self._ps[0]
        return all(p==p0 for p in self._ps)

    def _selections(self,n,genSelector):
        ''' returns a new Alea instance representing a probability distribution of
            the n-length tuples yielded by the given combinatorial generator
            genSelector, applied on the values of self distribution;
            the order of the elements of each built tuple is irrelevant: each tuple
            represents any permutation of its elements; the actual order of the
            elements of each tuple shall be the one defined by genSelector;
            assumes that n >= 0
            the efficient combinatorial algorithm is due to Paul Moore
        '''
        # First of all, get the values and weights for the distribution
        vps = dict(self.vps())
        # The total number of permutations of N samples is N!
        permutations = factorial(n)
        # We will calculate the frequency table for the result
        freqTable = []
        # Use genSelector to get the list of outcomes.
        # as itertools guarantees to give sorted output for sorted input,
        # giving the sorted sequence self._vs ensures our outputs are sorted
        for outcome in genSelector(self._vs,n):
            # We calculate the weight in 2 stages.
            # First we calculate the weight as if all values were equally
            # likely - in that case, the weight is N!/a!b!c!... where
            # a, b, c... are the sizes of each group of equal values
            weight = permutations
            # We run through the set counting and dividing as we go
            runLen = 0
            prevRoll = None
            for roll in outcome:
                if roll != prevRoll:
                    prevRoll = roll
                    runLen = 0
                runLen += 1
                if runLen > 1:
                    weight //= runLen
            # Now we take into account the relative weights of the values, by
            # multiplying the weight by the product of the weights of the
            # individual elements selected
            for roll in outcome:
                weight *= vps[roll]
            freqTable.append((outcome,weight))
        return Alea.fromValFreqs(*freqTable)

    def drawSortedWithReplacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITH replacement, whatever the order
            of drawing these elements; the returned values are tuples with n
            elements sorted by increasing order;
            assumes that n >= 0
            the efficient combinatorial algorithm is due to Paul Moore
        '''
        return self._selections(n,combinations_with_replacement)

    def drawSortedWithoutReplacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITHOUT replacement, whatever the order
            of drawing these elements; the returned values are tuples with n
            elements sorted by increasing order;
            assumes that 0 <= n <= number of values of self;
            note: if the probability distribution of self is uniform
            then the results is produced in an efficient way, tanks to the
            combinatorial algorithm of Paul Moore
        '''
        if self.isUniform():
            # the probability distribution is uniform,
            # the efficient algorithm of Paul Moore can be used
            return self._selections(n,combinations)
        else:
            # the probability distribution is not uniform,
            # we use the general algorithm less efficient:
            # make first a draw unsorted then sort (the sort makes the
            # required probability additions between permutations)
            return self.drawWithoutReplacement(n).map(lambda vs: tuple(sorted(vs))).getAlea()

    def drawWithReplacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITH replacement, taking the order
            of drawing into account; the returned values are tuples with n elements
            put in the order of their drawing;
            assumes that n >= 0
        '''
        if n == 0:
            return Lea.emptyTuple
        return self.map(makeTuple).times(n)

    def drawWithoutReplacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITHOUT replacement, taking the order
            of drawing into account; the returned values are tuples with n elements
            put in the order of their drawing
            assumes that n >= 0
            requires that n <= number of values of self, otherwise an exception
            is raised
        '''
        if n == 0:
            return Lea.emptyTuple
        if len(self._vs) == 1:
            if n > 1:
                raise Lea.Error("number of values to draw exceeds the number of possible values")
            return Alea(((self._vs[0],),),(1,))
        lcmP = calcLCM(self._ps)
        alea2s = tuple(Alea.fromValFreqsOrdered(*tuple((v0,p0) for (v0,p0) in self.vps() if v0 != v),reducing=False,check=False).drawWithoutReplacement(n-1) for v in self._vs)
        lcmP2 = calcLCM(alea2._count*(lcmP//p) for (alea2,p) in zip(alea2s,self._ps))
        f = lcmP2 // lcmP
        vps = []
        for (v,p,alea2) in zip(self._vs,self._ps,alea2s):
            g = (f*p) // alea2._count
            for (vt,pt) in alea2.vps():
                vps.append(((v,)+vt,g*pt))
        return Alea.fromValFreqsOrdered(*vps,reducing=False,check=False)

    @staticmethod
    def poisson(mean,precision):
        ''' static method, returns an Alea instance representing a Poisson probability
            distribution having the given mean; the distribution is approximated by
            the finite set of values that have probability > precision
            (i.e. low/high values with too small probabilities are dropped)
        '''
        precFactor = 0.5 / precision
        valFreqs = []
        p = exp(-mean)
        v = 0
        t = 0.
        while p > 0.0:
            valFreqs.append((v,int(0.5+p*precFactor)))
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
        valueStrings = tuple(str(v) for v in self._vs)
        ps = self._ps
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
            called on evaluation of "str(self)" and "repr(self)"
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

    def plot(self,title=None,fname=None,savefigArgs=dict(),**barArgs):
        ''' produces a matplotlib bar chart representing the probability distribution self
            with the given title (if not None); the bar chart may be customised by using
            named arguments barArgs, which are relayed to matplotlib.pyplot.bar function
            (see doc in http://matplotlib.org/api/pyplot_api.html)
            * if fname is None, then the chart is displayed on screen, in a matplotlib window;
              the previous chart, if any, is erased
            * otherwise, the chart is saved in a file specified by given fname as specified
              by matplotlib.pyplot.savefig; the file format may be customised by using
              savefigArgs argument, which is a dictionary relayed to matplotlib.pyplot.savefig
              function and containing named arguments expected by this function;
              example:
               flip.plot(fname='flip.png',savefigArgs=dict(bbox_inches='tight'),color='green')
            the method requires matplotlib package; an exception is raised if it is not installed
        '''
        try:
            plt
        except:
            raise Lea.Error("the plot() method requires the matplotlib package")
        if fname is None:
            # no file specified: erase the current chart, if any
            plt.clf()
        else:
            # file specified: switch off interactive mode
            plt.ioff()
        plt.bar(range(len(self._vs)),self.pmf(),tick_label=self._vs,align='center',**barArgs)
        plt.ylabel('Probability')
        if title is not None:
            plt.title(title)
        if fname is None:
            # no file specified: display the chart on screen
            plt.show()
        else:
            # file specified: save chart on file, using given parameters and switch back interactive mode
            plt.savefig(fname,**savefigArgs)
            plt.ion()

    def getAleaLeavesSet(self):
        ''' returns a set containing all the Alea leaves in the tree having the root self
            in the present case of Alea instance, it returns the singleton set with self as element
        '''
        return frozenset((self,))        
     
    def _getLeaChildren(self):
        # Alea instance has no children
        return ()

    def _clone(self,cloneTable):
        # note that the new Alea instance shares the immutable _vs and _ps attributes of self
        return Alea(self._vs,self._ps)

    def _getCount(self):
        ''' returns the total probability weight count (integer) of current Alea;
            this value depends on current binding (hence, the calculated value cannot be cached)
        '''
        if self._val is self:
            return self._count
        return 1

    def _genVPs(self):
        ''' generates tuples (v,p) where v is a value of self
            and p is the associated probability weight (integer > 0);
            the sequence follows the order defined on values
        '''
        return zip(self._vs,self._ps)

    vps = _genVPs

    def _genOneRandomMC(self):
        ''' generates one random value from the current probability distribution,
            WITHOUT precalculating the exact probability distribution (contrarily to 'random' method);
            this obeys the "binding" mechanism, so if the same variable is refered multiple times in
            a given expression, then same value will be yielded at each occurrence; 
            before yielding the random value v, this value v is bound to the current instance;
            then, if the current calculation requires to get again a random value on the current
            instance, then the bound value is yielded;
            the instance is rebound to a new value at each iteration, as soon as the execution
            is resumed after the yield;
            the instance is unbound at the end;
            the method calls the _genOneRandomMC method implemented in Lea subclasses;
        '''
        if self._val is not self:
            # distribution already bound to a value, because genOneRandomMC has been called already on self 
            # yield the bound value, in order to be consistent
            yield self._val
        else:
            try:
                # bind value v: this is important if an object calls genOneRandomMC on the same 
                # instance before resuming the present generator (see above)
                self._val = self.randomVal()
                # yield the bound value v
                yield self._val
            finally:
                # unbind value, after the random value has been bound or if an exception has been raised
                self._val = self
    
    def _p(self,val,checkValType=False):
        ''' returns the probability p/s of the given value val, as a tuple of naturals (p,s)
            where s is the sum of the probability weights of all values 
                  p is the probability weight of the given value val (from 0 to s)
            note: the ratio p/s is not reduced
            if checkValType is True, then raises an exception if some value in the
            distribution has a type different from val's
        '''
        p1 = 0
        if checkValType:
            errVal = self  # dumy value
            typeToCheck = type(val)
        # note: shall not exit the loop by a break/return (unbinding)
        for (v,p) in self._genVPs():
            if checkValType and not isinstance(v,typeToCheck):
                errVal = v
            if p1 == 0 and v == val:
                p1 = p
        if checkValType and errVal is not self:
            raise Lea.Error("found <%s> value although <%s> is expected"%(type(errVal).__name__,typeToCheck.__name__))
        return (p1,self._count)

    def cumul(self):
        ''' returns a list with the probability weights p that self <= value ;
            there is one element more than number of values; the first element is 0, then
            the sequence follows the order defined on values; if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned list is cached
        '''
        if len(self._cumul) == 1:
            cumulList = self._cumul
            pSum = 0
            for p in self._ps:
                pSum += p
                cumulList.append(pSum)
        return self._cumul

    def invCumul(self):
        ''' returns a tuple with the probability weights p that self >= value ;
            there is one element more than number of values; the first element is 0, then
            the sequence follows the order defined on values; if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned list is cached
        '''
        if len(self._invCumul) == 0:
            invCumulList = self._invCumul
            pSum = self._count
            for p in self._ps:
                invCumulList.append(pSum)
                pSum -= p
            invCumulList.append(0)
        return self._invCumul
            
    def randomVal(self):
        ''' returns a random value among the values of self, according to their probabilities
        '''
        return next(self._randomIter)
        
    def _createRandomIter(self):
        ''' generates an infinite sequence of random values among the values of self,
            according to their probabilities
        '''
        count = self._count
        probs = self.cumul()[1:]
        vals = self._vs
        while True:
            yield vals[bisect_right(probs,randrange(count))]    
        
    def randomDraw(self,n=None,sorted=False):
        ''' if n is None, returns a tuple with all the values of the distribution,
            in a random order respecting the probabilities
            (the higher count of a value, the most likely the value will be in the
             beginning of the sequence)
            if n > 0, then only n different values will be drawn
            if sorted is True, then the returned tuple is sorted
        '''
        if n is None:
           n = len(self._vs)
        elif n < 0:
            raise Lea.Error("randomDraw method requires a positive integer")    
        if n == 0:
            return ()
        lea1 = self
        res = []
        while True:
            lea1 = lea1.getAlea(sorting=False)
            x = lea1.random()
            res.append(x)
            n -= 1
            if n == 0:
                break
            lea1 = lea1.given(lea1!=x)
        if sorted:
            res.sort()
        return tuple(res)
    
    @memoize
    def pCumul(self,val):
        ''' returns, as an integer, the probability weight that self <= val
            note that it is not required that val is in the support of self
        '''
        return self.cumul()[bisect_right(self._vs,val)] 

    @memoize
    def pInvCumul(self,val):
        ''' returns, as an integer, the probability weight that self >= val
            note that it is not required that val is in the support of self
        '''
        return self.invCumul()[bisect_left(self._vs,val)] 

    @staticmethod
    def fastExtremum(cumulFunc,*aleaArgs):
        ''' static method, returns a new Alea instance giving the probabilities
            to have the extremum value (min or max) of each combination of the
            given Alea args;
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
            for (v,p) in aleaArg1.vps():
                valFreqsDict[v] = p * cumulFunc(aleaArg2,v)
            for (v,p) in aleaArg2.vps():
                valFreqsDict[v] += (cumulFunc(aleaArg1,v)-aleaArg1._p(v)[0]) * p
            return Lea.fromValFreqsDict(valFreqsDict)
        return Alea.fastExtremum(cumulFunc,aleaArgs[0],Alea.fastExtremum(cumulFunc,*aleaArgs[1:]))

    # WARNING: the following methods are called without parentheses (see Lea.__getattr__)

    indicatorMethodNames = ('P','Pf','mean','var','std','mode','entropy','information')

    def P(self):
        ''' returns a ProbFraction instance representing the probability of True,
            from 0/1 to 1/1;
            raises an exception if some value in the distribution is not boolean
            (this is NOT the case with self.p(True))
            WARNING: this method is called without parentheses
        '''
        return ProbFraction(*self._p(True,checkValType=True))

    def Pf(self):
        ''' returns the probability of True, as a floating point number,
            from 0.0 to 1.0;
            raises an exception if some value in the distribution is not boolean
            (this is NOT the case with self.pmf(True))
            WARNING: this method is called without parentheses
        '''
        return float(self.P)
        
    def mean(self):
        ''' returns the mean value of the probability distribution, which is the
            probability weighted sum of the values;
            requires that
            1 - the values can be subtracted together,
            2 - the differences of values can be multiplied by integers,
            3 - the differences of values multiplied by integers can be
                added to the values,
            4 - the sum of values calculated in 3 can be divided by a float
                or an integer;
            if any of these conditions is not met, then the result depends of the
            value class implementation (likely, raised exception)
            WARNING: this method is called without parentheses
        '''
        res = None
        x0 = None
        for (x,p) in self.vps():
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
                # if the / operator is not supported with float as denominator
                # e.g. datetime.timedelta in Python 2.x 
                x0 += res / self._count    
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
        for (v,p) in self.vps():
            res += p*(v-m)**2
        return res / float(self._count)    

    def std(self):
        ''' returns a float number representing the standard deviation of the probability distribution
            requires that the requirements of the variance() method are met
            WARNING: this method is called without parentheses
        '''      
        return sqrt(self.var)
 
    def mode(self):
        ''' returns a tuple with the value(s) of the probability distribution having the highest probability 
            WARNING: this method is called without parentheses
        '''
        maxP = max(self._ps)
        return tuple(v for (v,p) in self.vps() if p == maxP)
            
    def entropy(self):
        ''' returns a float number representing the entropy of the probability distribution
            WARNING: this method is called without parentheses
        '''
        res = 0
        count = float(self._count)
        for (v,p) in self.vps():
            if p > 0:
               p /= count
               res -= p*log(p)
        return res / LOG2

    def internal(self,indent='',refs=None):
        ''' returns a string representing the inner definition of self;
            if the same lea child appears multiple times, it is expanded only
            on the first occurrence, the other ones being marked with
            reference id; the arguments are used only for recursive calls
            from Lea.internal method, they can be ignored for a normal usage
        '''
        if refs is None:
            refs = set()
        if self in refs:
            return self._id()+'*'
        refs.add(self)
        return self._id() + str(tuple(self.vps()))
