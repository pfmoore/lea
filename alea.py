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
from .prob_number import ProbNumber
from .prob_fraction import ProbFraction
from .prob_decimal import ProbDecimal
from fractions import Fraction
from decimal import Decimal
from random import random
from bisect import bisect_left, bisect_right
from itertools import combinations, combinations_with_replacement
from math import exp, factorial
from .toolbox import log2, memoize, zip, next, dict, defaultdict, makeTuple

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
    An Alea instance is defined by given value-probability pairs. The probabilities can be
    expressed as any object with arithmetic semantic. The main candidates are float, fraction
    or symbolic expressions. Values having null probability counters are dropped.
    '''

    __slots__ = ('_vs','_ps','_cumul','_invCumul','_randomIter','_cachesByFunc')

    @staticmethod
    def _simplify(p,toFloat=False):
        if hasattr(p,'factor'):
            p = p.factor()
        elif toFloat:
            p = float(p)
        return p

    def __init__(self,vs,ps,normalization=True):
        ''' initializes Alea instance's attributes
            vs is a sequence of values
            ps is a sequence of probabilities (same size and order as ps)
            if normalization argument is True (default), then each element
            of the given ps is divided by the sum of all ps before being stored
            (in such case, it's not mandatory to have true probabilities for ps
             elements; these could be simple counters for example)
        '''
        Lea.__init__(self)
        # for an Alea instance, the alea cache is itself
        self._alea = self
        self._vs = tuple(vs)
        if normalization:
            pSum = sum(ps)
            if hasattr(pSum,'factor'):
                self._ps = tuple((p/pSum).factor() for p in ps)
            else:
                self._ps = tuple(p/pSum for p in ps)
        else:
            self._ps = tuple(ps)
        self._cumul = [0]
        self._invCumul = []
        self._randomIter = self._createRandomIter()
        self._cachesByFunc = dict()

    '''
    @staticmethod
    def setProbType(probTypeCode):
        # TODO remove checkProbability
        Alea.checkProbability = True
        if probTypeCode == 'f':
            probType = float
        elif probTypeCode == 'r':
            from .prob_fraction import ProbFraction
            probType = ProbFraction
        elif probTypeCode == 'd':
            from .prob_decimal import ProbDecimal
            probType = ProbDecimal
        elif probTypeCode == 's':
            from sympy import factor #as simplifyFunc
            from sympy import symbols, Symbol
            class ProbSymbol(Symbol):
                def __new__(cls, s=0):
                    if isinstance(s,str):
                        return Symbol(s)
                    return s
                def simplify(self,toFloat=False):
                    return factor(self)
            probType = ProbSymbol
            Alea.checkProbability = False
        else:
            raise Lea.Error("unknown probability type code '%s', should be 'f', 'r', 'd' or 's'"%probTypeCode)
        Alea.init(probType)
        Alea.probTypeCode = probTypeCode
    '''


    @staticmethod
    def setProbType(probTypeCode):
        # TODO remove checkProbability
        Alea.checkProbability = True
        if probTypeCode == 'f':
            probType = float
        elif probTypeCode == 'r':
            probType = Fraction
        elif probTypeCode == 'd':
            probType = Decimal
        elif probTypeCode == 's':
            from sympy import factor #as simplifyFunc
            from sympy import symbols, Symbol
            class ProbSymbol(Symbol):
                def __new__(cls, s=0):
                    if isinstance(s,str):
                        return Symbol(s)
                    return s
                #def simplify(self,toFloat=False):
                #    return factor(self)
            probType = ProbSymbol
            Alea.checkProbability = False
        else:
            raise Lea.Error("unknown probability type code '%s', should be 'f', 'r', 'd' or 's'"%probTypeCode)
        Alea.init(probType)
        Alea.probTypeCode = probTypeCode

    # Constants representing certain values (Alea static attributes)
    true = None
    false = None
    zero = None
    emptyTuple = None

    probTypeCode = None
    _probOne = None
    _probType = None

    @staticmethod
    def init(probType):
        Alea._probOne = probType(1)
        Alea._probType = probType
        Alea.true  = Alea.coerce(True)
        Alea.false = Alea.coerce(False)
        Alea.zero  = Alea.coerce(0)
        Alea.emptyTuple = Alea.coerce(())

    # constructor methods
    # -------------------

    @staticmethod
    def coerce(value):
        ''' static method, returns a Lea instance corresponding the given value:
            if the value is a Lea instance, then it is returned
            otherwise, an Alea instance is returned, with given value
            as unique value, with a probability of 1.
        '''
        if not isinstance(value,Lea):
            #return Alea((value,),(Alea._probOne,),normalization=False)
            return Alea((value,),(1,),normalization=False)
        return value

    def new(self):
        ''' returns a new Alea instance which is an independent shallow copy of self;
            note that the present method overloads Lea.new to be more efficient
        '''
        # note that the new Alea instance shares the immutable _vs and _ps attributes of self
        newAlea = Alea(self._vs,self._ps,normalization=False)
        # it can share also the mutable _cumul and _invCumul attributes of self (lists)
        newAlea._cumul = self._cumul
        newAlea._invCumul = self._invCumul
        return newAlea

    '''
    Not used in Lea 3
    @staticmethod
    def fromValFreqsDictGen(probDict):
        '' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob}, where prob is an integer number,
            a floating-point number or a fraction (Fraction or ProbFraction instance)
            so that each value val has probability proportional to prob to occur
            any value with null probability is ignored (hence not stored)
            the values are sorted if possible (i.e. no exception on sort), 
            otherwise, the order of values is unspecified; 
            if the sequence is empty, then an exception is raised
        ''
        probFractions = tuple(ProbFraction.coerce(prob) for prob in probDict.values())
        # TODO Check positive
        probWeights = ProbFraction.getProbWeights(probFractions)
        return Alea.fromValFreqsDict(dict(zip(probDict.keys(),probWeights)))
    '''


    __contructorArgNames = frozenset(('ordered','sorting','normalization','check','frac'))

    @staticmethod
    def _parsedKwargs(kwargs):
        ''' return (ordered,sorting,normalization,check) tuple, with values found
            in the given kwargs dictionary (keywords); for missing keywords,
            the default values are False, True, True, True, respectively, except
            if ordered=True and sorting is missing, then sorting=False;
            requires that the given kwargs dictionary contains no other keywords
            than those defined above;
            requires that ordered and sorting are not set to True together
        '''
        argNames = frozenset(kwargs.keys())
        unknownArgNames = argNames - Alea.__contructorArgNames
        if len(unknownArgNames) > 0:
            raise Lea.Error("unknown argument keyword '%s'; shall be only among %s"%(next(iter(unknownArgNames)),tuple(Alea.__contructorArgNames)))
        ordered = kwargs.get('ordered',False)
        normalization = kwargs.get('normalization',True)
        check = kwargs.get('check',True)
        if ordered and 'sorting' not in kwargs:
            sorting = False
        else:
            sorting = kwargs.get('sorting',True)
            if ordered and sorting:
                raise Lea.Error("ordered and sorting arguments cannot be set to True together")
        return (ordered,sorting,normalization,check)

    '''
    # TODO: remove tests for symbolic calculations
    @staticmethod
    def _getVPsIter(vps,normalization):
        #if any(p < 0 for (_,p) in vps):
        #    raise Lea.Error("negative probability")
        probSum = sum(p for (_,p) in vps)
        if probSum == 0:
            raise Lea.Error("impossible to build a probability distribution with no value")
        if normalization and probSum != 1:
            # return ((v,p/probSum) for (v,p) in vps if p > 0)
            return ((v,p/probSum) for (v,p) in vps)
        print (normalization, probSum)
        return iter(vps)
    '''

    @staticmethod
    def _fromValFreqsDict(probDict,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob};
            if the sequence is empty, then an exception is raised;
            the method admits two optional boolean argument (in kwargs):
            * sorting (default:True): if True, then the values for displaying
            the distribution or getting the values will be sorted if possible
            (i.e. no exception on sort); otherwise, or if sorting=False, then
            the order of values is unspecified; 
            * normalization (default:True): if True, then each element
            of the given ps is divided by the sum of all ps before being stored
            (in such case, it's not mandatory to have true probabilities for ps
             elements; these could be simple counters for example)
        '''
        (ordered,sorting,normalization,check) = Alea._parsedKwargs(kwargs)
        if ordered:
            raise Lea.Error("cannot keep order of dictionary keys")
        vpsIter = probDict.items()
        if sorting:
            vps = list(vpsIter)
            try:            
                vps.sort()
            except:
                # no ordering relationship on values (e.g. complex numbers)
                pass
        else:
            vps = vpsIter
        return Alea(*zip(*vps),normalization=normalization)

    @staticmethod
    def fromValFreqsDict(probDict,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob};
            if the sequence is empty, then an exception is raised;
            the method admits two optional boolean argument (in kwargs):
            * sorting (default:True): if True, then the values for displaying
            the distribution or getting the values will be sorted if possible
            (i.e. no exception on sort); otherwise, or if sorting=False, then
            the order of values is unspecified;
            * normalization (default:True): if True, then each element
            of the given ps is divided by the sum of all ps before being stored
            (in such case, it's not mandatory to have true probabilities for ps
             elements; these could be simple counters for example)
        '''
        probType = Alea._probType
        return Alea._fromValFreqsDict(dict((v,probType(p)) for (v,p) in valueFreqs),**kwargs)

    @staticmethod
    def fromVals(*vals,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of values, so that each value occurrence is
            taken as equiprobable; if each value occurs exactly once, then the
            distribution is uniform, i.e. the probability of each value is equal
            to 1 / #values; otherwise, the probability of each value is equal
            to its frequency in the sequence;
            the following optional arguments (kwargs) are expected:
             'frac', 'ordered', 'sorting', 'normalization','check';
            * if frac is False (default) then  the probabilities are stored as
            floating-point numbers;
            * if frac is True, then these are stored as fractions (instances
            of ProbFraction);
            for treatment of other optional arguments, see doc of
            Alea.fromValFreqsDict static method;
            if the sequence is empty, then an exception is raised;
        '''
        # TODO: could add decimal
        #frac = kwargs.get('frac',False)
        #one = ProbFraction.one if frac else 1.0
        #probOne = Alea._probOne
        return Alea.fromValFreqs(*((val,1) for val in vals),**kwargs)

    @staticmethod
    def fromSeq(vals,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of values (e.g. a list, tuple, iterator,...);
            this is a convenience method, equivalent to
              Alea.fromVals(*vals,**kwargs)
            for detailed description, refer to the doc of this method
        '''
        return Alea.fromVals(*vals,**kwargs)
    '''
    @staticmethod
    def fromValFreqs(*valueFreqs,**kwargs):
        '' static method, returns an Alea instance representing a distribution
            for the given sequence of (v,p) tuples, where p is the
            probability of v or some number proportional to this probability;
            if the same v occurs multiple times, then the associated p are summed
            together;
            if the sequence is empty, then an exception is raised;
            for treatment of optional kwargs keywords arguments, see doc of
            Alea.fromValFreqsDict;
        ''
        (ordered,sorting,normalization,check) = Alea._parsedKwargs(kwargs)
        if ordered:
            return Alea._fromValFreqsOrdered(*valueFreqs,**kwargs)
        probType = Alea._probType
        probDict = defaultdict(probType)
        for (value,freq) in valueFreqs:
            probDict[value] += probType(freq)
        return Alea.fromValFreqsDict(probDict,**kwargs)
    '''

    @staticmethod
    def fromValFreqs(*valueFreqs,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (v,p) tuples, where p is the
            probability of v or some number proportional to this probability;
            if the same v occurs multiple times, then the associated p are summed
            together;
            if the sequence is empty, then an exception is raised;
            for treatment of optional kwargs keywords arguments, see doc of
            Alea.fromValFreqsDict;
        '''
        probType = Alea._probType
        return Alea._fromValFreqs(*((v,probType(p)) for (v,p) in valueFreqs),**kwargs)

    @staticmethod
    def _fromValFreqs(*valueFreqs,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (v,p) tuples, where p is the
            probability of v or some number proportional to this probability;
            if the same v occurs multiple times, then the associated p are summed
            together;
            if the sequence is empty, then an exception is raised;
            for treatment of optional kwargs keywords arguments, see doc of
            Alea.fromValFreqsDict;
        '''
        (ordered,sorting,normalization,check) = Alea._parsedKwargs(kwargs)
        if ordered:
            return Alea._fromValFreqsOrdered(*valueFreqs, **kwargs)
        probDict = defaultdict(int)
        for (value,freq) in valueFreqs:
            probDict[value] += freq
        return Alea._fromValFreqsDict(probDict,**kwargs)

    '''
    @staticmethod
    def fromValFreqsDictArgs(**probDict):
        '' static method, same as fromValFreqsDict, excepting that the dictionary
            is passed in a **kwargs style
        ''
        return Alea.fromValFreqsDict(probDict)
    '''

    @staticmethod
    def _fromValFreqsOrdered(*valueFreqs, **kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural
            number so that each value is taken with the given frequency
            the frequencies are reduced by dividing them by their GCD;
            the values will be stored and displayed in the given order (no sort);
            if the sequence is empty, then an exception is raised;
            requires that each value has a unique occurrence;
            the method admits 2 optional boolean argument (kwargs):
            * check (default: True): if True and if a value occurs multiple
            times, then an exception is raised;        
        '''
        (ordered,sorting,normalization,check) = Alea._parsedKwargs(kwargs)
        (vs,ps) = zip(*valueFreqs)
        # check duplicates
        if check and len(frozenset(vs)) < len(vs):
            raise Lea.Error("duplicate values")
        return Alea(vs,ps,normalization)

    checkProbability = True

    @staticmethod
    def _checkProb(p):
        if Alea.checkProbability:
            if p < 0:
                raise Lea.Error("negative probability")
            if p > 1:
                raise Lea.Error("probability strictly greater than 1")

    @staticmethod
    def _binaryDistribution(p1,v1,v2):
        # TODO: check this
        p1 = Alea._probType(p1)
        Alea._checkProb(p1)
        if p1 == 1:
            (vs,ps) = ((v1,),(1,))
        elif p1 == 0:
            (vs,ps) = ((v2,),(1,))
        else:
            #p1 = Alea._probType(p1)
            (vs,ps) = ((v1,v2),(p1,Alea._probOne-p1))
        return Alea(vs,ps,normalization=False)

    @staticmethod
    def boolProb(p):
        return Alea._binaryDistribution(p,True,False)

    @staticmethod
    def bernoulli(p):
        ''' static method, returns an Alea instance representing a bernoulli
            distribution giving 1 with probability pNum/pDen and 0 with
            complementary probability;
            if pDen is None, then pNum expresses the probability as a float,
            a string, a Python's Fraction or a Decimal instance, in the same
            way as Fraction constructor; for strings, percentages are also
            allowed using the '%' suffix
        '''
        return Alea._binaryDistribution(p,1,0)

    @staticmethod
    def binom(n,p):
        ''' static method, returns an Alea instance representing a binomial
            distribution giving the number of successes among a number n of
            independent experiments, each having probability pNum/pDen of success;
            if pDen is None, then pNum expresses the probability as a float,
            a string, a Python's Fraction or a Decimal instance, in the same
            way as Fraction constructor; for strings, percentages are also
            allowed using the '%' suffix
        '''
        return Alea.bernoulli(p).times(n)

    @staticmethod
    def interval(fromVal,toVal,frac=False):
        ''' static method, returns an Alea instance representing a uniform probability
            distribution, for all the integers in the interval [fromVal,toVal]
        '''
        return Alea.fromVals(*range(fromVal,toVal+1),frac=frac)

    @staticmethod
    def fromCSVFilename(csvFilename,colNames=None,dialect='excel',**fmtparams):
        ''' static method, returns an Alea instance representing the joint probability
            distribution of the data read in the CSV file of the given csvFilename;
            it is similar to Alea.fromCSVFile method, except that it takes a filename
            instead of an open file (i.e. the method opens itself the file for reading);
            see Alea.fromCSVFile doc for more details
        '''
        (attrNames,dataFreq) = readCSVFilename(csvFilename,colNames,dialect,**fmtparams)
        return Alea.fromValFreqs(*dataFreq).asJoint(*attrNames)

    @staticmethod
    def fromCSVFile(csvFile,colNames=None,dialect='excel',**fmtparams):
        ''' static method, returns an Alea instance representing the joint probability
            distribution of the data read in the given CSV file;
            the arguments follow the same semantics as those of Python's csv.reader
            method, which supports different CSV formats;
            see doc in https://docs.python.org/2/library/csv.html
            * if colNames is None, then the fields found in the first read row of the CSV
              file provide information on the attributes: each field is made up of a name,
              which shall be a valid identifier, followed by an optional 3-characters type
              code among
                {b} -> boolean
                {i} -> integer
                {f} -> float
                {s} -> string
                {#} -> count
              if the type code is missing for a given field, the type string is assumed for
              this field; for example, using the comma delimiter (default), the first row
              in the CSV file could be:
                  name,age{i},heigth{f},married{b}
            * if colNames is not None, then colNames shall be a sequence of strings giving
              attribute information as described above, e.g.
                  ('name','age{i}','heigth{f}','married{b}')
              it assumed that there is NO header row in the CSV file
            the type code defines the conversion to be applied to the fields read on the
            data lines; if the read value is empty, then it is converted to Python's None,
            except if the type is string, then, the value is the empty string;
            if the read value is not empty and cannot be parsed for the expected type, then
            an exception is raised; for boolean type, the following values (case
            insensitive):
              '1', 't', 'true', 'y', 'yes' are interpreted as Python's True,
              '0', 'f', 'false', 'n', 'no' are interpreted as Python's False;
            the {#} code identifies a field that provides a count number of the row,
            representing the probability of the row or its frequency as a positive integer;
            such field is NOT included as attribute of the joint distribution; it is useful
            to define non-uniform probability distribution, as alternative to repeating the
            same row multiple times
        '''
        (attrNames,dataFreq) = readCSVFile(csvFile,colNames,dialect,**fmtparams)
        return Alea.fromValFreqs(*dataFreq).asJoint(*attrNames)

    @staticmethod
    def fromPandasDF(dataframe,indexColName=None):
        ''' static method, returns an Alea instance representing the joint probability
            distribution from the given pandas dataframe;
            the attribute names of the distribution are those of the column of the
            given dataframe; the first field in each item of the dataframe is assumed
            to be the index; its treatment depends on given indexColName:
            if indexColName is None, then this index field is ignored
            otherwise, it is put in the joint distribution with indexColName as
            attribute name
        '''
        # TODO: retrieve index_col in df, if not 0
        attrNames = tuple(dataframe.columns)
        if indexColName is None:
            valuesIter = (t[1:] for t in dataframe.itertuples())
            attrNames = dataframe.columns
        else:
            valuesIter = dataframe.itertuples()
            attrNames = (indexColName,) + attrNames
        return Alea.fromSeq(valuesIter).asJoint(*attrNames)

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
            return Alea.emptyTuple
        return self.map(makeTuple).times(n)

    '''
    def drawWithoutReplacement(self,n):
        '' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITHOUT replacement, taking the order
            of drawing into account; the returned values are tuples with n elements
            put in the order of their drawing
            assumes that n >= 0
            requires that n <= number of values of self, otherwise an exception
            is raised
        ''
        if n == 0:
            return Alea.emptyTuple
        if len(self._vs) == 1:
            if nbValues > 1:
                raise Lea.Error("number of values to draw exceeds the number of possible values")
            return Alea(((self._vs[0],),),(1,))
        lcmP = calcLCM(self._ps)
        alea2s = tuple(Alea._fromValFreqsOrdered(*tuple((v0,p0) for (v0,p0) in self.vps() if v0 != v),normalization=False,check=False).drawWithoutReplacement(n-1) for v in self._vs)
        lcmP2 = calcLCM(alea2._count*(lcmP//p) for (alea2,p) in zip(alea2s,self._ps))
        f = lcmP2 // lcmP
        vps = []
        for (v,p,alea2) in zip(self._vs,self._ps,alea2s):
            g = (f*p) // alea2._count
            for (vt,pt) in alea2.vps():
                vps.append(((v,)+vt,g*pt))
        return Alea._fromValFreqsOrdered(*vps,normalization=False,check=False)
    '''

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
            return Alea.emptyTuple
        if len(self._vs) == 1:
            if n > 1:
                raise Lea.Error("number of values to draw exceeds the number of possible values")
            #return Alea(((self._vs[0],),),(1,))
            return Alea.coerce((self._vs[0],))
        alea2s = tuple(Alea._fromValFreqsOrdered(*tuple((v0, p0) for (v0, p0) in self.vps() if v0 != v), check=False).drawWithoutReplacement(n - 1) for v in self._vs)
        vps = []
        for (v,p,alea2) in zip(self._vs,self._ps,alea2s):
            for (vt,pt) in alea2.vps():
                vps.append(((v,)+vt,p*pt))
        return Alea._fromValFreqsOrdered(*vps, check=False)
    '''

    @staticmethod
    def poisson(mean,precision):
        '' static method, returns an Alea instance representing a Poisson probability
            distribution having the given mean; the distribution is approximated by
            the finite set of values that have probability > precision
            (i.e. low/high values with too small probabilities are dropped)
        ''
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
    '''

    def poisson(mean,precision):
        ''' static method, returns an Alea instance representing a Poisson probability
            distribution having the given mean; the distribution is approximated by
            the finite set of values that have probability > precision
            (i.e. low/high values with too small probabilities are dropped)
        '''
        valFreqs = []
        p = exp(-mean)
        v = 0
        t = 0.
        while p >= precision or v <= mean:
            if p >= precision:
                valFreqs.append((v,p))
            t += p
            v += 1
            p = (p*mean) / v
        return Alea.fromValFreqs(*valFreqs)

    def asString(self,kind='.',nbDecimals=None,histoSize=100):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability in a format depending of given kind, which is string among
            '.', '%', '-', '/-', '.-', '%-';
            the probabilities are displayed as
            - if kind[0] is '.' : decimals with given nbDecimals digits
            - if kind[0] is '%' : percentage decimals with given nbDecimals digits
            - if kind[0] is '-' : histogram bar made up of repeated '-', such that
                                  a bar length of histoSize represents a probability 1 
            if kind[1] is '-', the histogram bars with '-' are appended after 
                               numerical representation of probabilities
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used
        '''
        if kind not in ('.', '%', '-', '/-', '.-', '%-'):
            raise Lea.Error("invalid display format '%s'"%kind)
        valueStrings = tuple(str(v) for v in self._vs)
        ps = self._ps
        vm = max(len(v) for v in valueStrings)
        linesIter = (v.rjust(vm)+' : ' for v in valueStrings)
        probRepresentation = kind[0]
        withHisto = kind[-1] == '-'
        if probRepresentation == '.':
            if nbDecimals is None:
                fmt = "%s%s"
            else:
                fmt = "%%s%%.%df" % nbDecimals
            linesIter = (fmt%(line,p) for (line,p) in zip(linesIter,ps))
        elif probRepresentation == '%':
            if nbDecimals is None:
                fmt = "%s%s %%"
            else:
                fmt = "%%s%%%d.%df %%%%" % (4+nbDecimals,nbDecimals)
            linesIter = (fmt%(line,100.*p) for (line,p) in zip(linesIter,ps))
        if withHisto:
            linesIter = (line+' '+int(0.5+(p)*histoSize)*'-' for (line,p) in zip(linesIter,ps))
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

    def withFloatProb(self):
        return Alea(self._vs,(float(p) for p in self._ps),normalization=False)

    def asPct(self,nbDecimals=2):
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
        return Alea(self._vs,self._ps,normalization=False)

    '''
    def _getCount(self):
        '' returns the total probability weight count (integer) of current Alea;
            this value depends on current binding (hence, the calculated value cannot be cached)
        ''
        if self._val is self:
            return self._count
        return 1
    '''

    def _genVPs(self):
        ''' generates tuples (v,p) where v is a value of self
            and p is the associated probability;
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
    
    '''
    def _p(self,val,checkValType=False):
        '' returns the probability p of the given value val
            if checkValType is True, then raises an exception if some value in the
            distribution has a type different from val's
        ''
        p1 = 0
        if checkValType:
            errVal = self  # dummy value
            typeToCheck = type(val)
        # note: shall not exit the loop by a break/return (unbinding)
        for (v,p) in self._genVPs():
            if checkValType and not isinstance(v,typeToCheck):
                errVal = v
            if p1 == 0 and v == val:
                p1 = p
        if checkValType and errVal is not self:
            raise Lea.Error("found <%s> value although <%s> is expected"%(type(errVal).__name__,typeToCheck.__name__))
        return p1
    '''

    def _p(self,val,checkValType=False):
        ''' returns the probability p of the given value val
            if checkValType is True, then raises an exception if some value in the
            distribution has a type different from val's
        '''
        p1 = None
        if checkValType:
            errVal = self  # dummy value
            typeToCheck = type(val)
        for (v,p) in self._genVPs():
            if checkValType and not isinstance(v,typeToCheck):
                errVal = v
            if p1 is None and v == val:
                p1 = p
        if checkValType and errVal is not self:
            raise Lea.Error("found <%s> value although <%s> is expected"%(type(errVal).__name__,typeToCheck.__name__))
        if p1 is None:
            # val is absent form self: the probability is null, casted in the type of the last probability found
            p1 = 0 * p
        return p1

    def cumul(self):
        ''' returns a list with the probabilities p that self <= value ;
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
        ''' returns a tuple with the probabilities p that self >= value ;
            there is one element more than number of values; the first element is 0, then
            the sequence follows the order defined on values; if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned list is cached
        '''
        if len(self._invCumul) == 0:
            invCumulList = self._invCumul
            pSum = 1
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
        probs = self.cumul()[1:]
        vals = self._vs
        while True:
            yield vals[bisect_right(probs,random())]
        
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
        ''' returns, as an integer, the probability that self <= val
            note that it is not required that val is in the support of self
        '''
        return self.cumul()[bisect_right(self._vs,val)] 

    @memoize
    def pInvCumul(self,val):
        ''' returns, as an integer, the probability that self >= val
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
                valFreqsDict[v] += (cumulFunc(aleaArg1,v)-aleaArg1._p(v)) * p
            return Alea.fromValFreqsDict(valFreqsDict)
        return Alea.fastExtremum(cumulFunc,aleaArgs[0],Alea.fastExtremum(cumulFunc,*aleaArgs[1:]))

    # WARNING: the following methods are called without parentheses (see Lea.__getattr__)

    indicatorMethodNames = ('P','Pf','mean','meanF','var','varF','std','stdF','mode','entropy','information')

    _upcastProbClass = {Fraction : ProbFraction,
                        Decimal  : ProbDecimal }

    def P(self):
        ''' returns the probability of True, expressed in the type used in self;
            raises an exception if some value in the distribution is not boolean
            (this is NOT the case with self.p(True))
            WARNING: this method is called without parentheses
        '''
        p = self._p(True,checkValType=True)
        upcastProbClass = Alea._upcastProbClass.get(p.__class__)
        if upcastProbClass is not None:
            p = upcastProbClass(p)
        return p

    def Pf(self):
        ''' returns the probability of True, expressed as a float between 0.0 and 1.0;
            raises an exception if some value in the distribution is not boolean
            (this is NOT the case with self.p(True))
            WARNING: this method is called without parentheses
        '''
        return float(self._p(True,checkValType=True))

    def _mean(self):
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
        '''
        res = None
        v0 = None
        for (v,p) in self.vps():
            if v0 is None:
                v0 = v
            elif res is None:
                res = p * (v-v0)
            else:
                res += p * (v-v0)
        if res is not None:
            v0 += res
        return v0

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
        return Alea._simplify(self._mean(),False)

    def meanF(self):
        return Alea._simplify(self._mean(),True)

    def _var(self):
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
        m = self._mean()
        for (v,p) in self.vps():
            res += p*(v-m)**2
        return res

    def var(self):
        return Alea._simplify(self._var(),False)

    def varF(self):
        return Alea._simplify(self._var(),True)

    def _std(self):
        ''' returns a float number representing the standard deviation of the probability distribution
            requires that the requirements of the variance() method are met
            WARNING: this method is called without parentheses
        '''
        var = self._var()
        sqrtExp = var.__class__(0.5)
        return self._var() ** sqrtExp

    def std(self):
        return Alea._simplify(self._std(),False)

    def stdF(self):
        return Alea._simplify(self._std(),True)

    def mode(self):
        ''' returns a tuple with the value(s) of the probability distribution having the highest probability 
            WARNING: this method is called without parentheses
        '''
        maxP = max(self._ps)
        return tuple(v for (v,p) in self.vps() if p == maxP)

    def informationOf(self,val):
        ''' returns a float number representing the information of given val,
            expressed in bits
            raises an exception if given val is impossible
        '''
        p = self._p(val)
        try:
            if p == 0:
                raise Lea.Error("no information from impossible value")
            return -log2(p)
        except TypeError:
            try:
                from sympy import log
                return -log(p,2)
            except:
                raise Lea.Error("cannot calculate logarithm of %s"%p)


    def entropy(self):
        ''' returns a float number representing the entropy of the probability distribution
            WARNING: this method is called without parentheses
        '''
        res = 0
        try:
            for (v,p) in self.vps():
                if p > 0:
                    res -= p*log2(p)
            return res
        except TypeError:
            try:
                from sympy import log
                for (v,p) in self.vps():
                    res -= p*log(p)
                return res / log(2)
            except:
                raise Lea.Error("cannot calculate logarithm on given probability types")

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


