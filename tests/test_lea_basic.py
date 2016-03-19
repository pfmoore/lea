from lea import Lea
from lea import ProbFraction as PF
from fractions import Fraction
from decimal import Decimal
import pytest
import math
import sys

def test_clone():
    dist1 = Lea.fromVals(1,2,3,4)
    dist2 = dist1.clone()
    assert dist1.equiv(dist2)
    assert dist1 is not dist2

def test_id():
    dist1 = Lea.fromVals(1,2,3,4)
    dist2 = dist1.clone()
    assert dist1._id() != dist2._id()
    assert isinstance(dist1._id(), str)

def test_get_alea_leaves():
    dist1 = Lea.fromVals(1,2,3,4)
    dist2 = Lea.fromVals(2,4,6,8)
    distcalc = (dist1 + dist2) * (dist1 - dist2)
    assert distcalc.getAleaLeavesSet() == {dist1, dist2}

# Constructors
def test_fromvals():
    d = Lea.fromVals(1,2, ordered=True, sorting=False, reducing=False)

def test_fromvals_errors():
    # Must be at least one value
    with pytest.raises(Lea.Error):
        d = Lea.fromVals()
    # No invalid keyword args
    with pytest.raises(Lea.Error):
        d = Lea.fromVals(1,2, foo=3)
    # Cannot have both ordered and sorting
    with pytest.raises(Lea.Error):
        d = Lea.fromVals(1,2, ordered=True, sorting=True)
    # Cannot have duplicates with ordered
    with pytest.raises(Lea.Error):
        d = Lea.fromVals(1,1, ordered=True)
    # Cannot have ordered with dictionary
    with pytest.raises(Lea.Error):
        d = Lea.fromValFreqsDict({1: 2, 2: 5}, ordered=True)

def test_fromvals_ordered():
    d = Lea.fromVals(2,1,3, ordered=True)
    assert tuple(d.vals()) == (2,1,3)
    d = Lea.fromValFreqs((2,9),(1,7),(3,5), ordered=True)
    assert tuple(d.vals()) == (2,1,3)

def test_fromvals_sorting():
    d = Lea.fromVals(2,1,3,2, sorting=True)
    assert tuple(d.vals()) == (1,2,3)

def test_fromvals_reducing():
    d = Lea.fromVals(1,1,2,2, reducing=False)
    assert tuple(d.vps()) == ((1,2), (2,2))

def test_from_dict():
    d = Lea.fromValFreqsDict({'a': 5, 'b': 6})
    assert set(d.vps()) == {('a', 5), ('b', 6)}
    d = Lea.fromValFreqsDictArgs(a=5, b=6)
    assert set(d.vps()) == {('a', 5), ('b', 6)}

def test_boolprob():
    d = Lea.boolProb(0)
    assert d.p(True) == PF(0)
    d = Lea.boolProb(1,2)
    assert d.p(True) == PF(1,2)
    d = Lea.boolProb(1)
    assert d.p(True) == PF(1)
    d = Lea.boolProb(Fraction(123456789,1000000000))
    assert d.p(True) == PF(123456789,1000000000)
    d = Lea.boolProb(Decimal("0.123456789"))
    assert d.p(True) == PF(123456789,1000000000)
    d = Lea.boolProb("12.3456789%")
    assert d.p(True) == PF(123456789,1000000000)
    assert d.p(False) == 1 - PF(123456789,1000000000)
    d = Lea.boolProb(0.0)
    assert d.p(True) == PF(0)
    d = Lea.boolProb(0.5)
    assert d.p(True) == PF(1,2)
    d = Lea.boolProb(1.0)
    assert d.p(True) == PF(1)
    # the following cases cannot assert exact values due to float representation
    # the pmf method shall convert the prob. fraction back into the given float (no loss of precision)
    d = Lea.boolProb(0.2)
    assert d.pmf(True) == 0.2
    assert d.pmf(False) == 1.0 - 0.2
    d = Lea.boolProb(0.123456789)
    assert d.pmf(True) == 0.123456789
    assert d.pmf(False) == 1.0 - 0.123456789
    d = Lea.boolProb(sys.float_info.epsilon)
    assert d.pmf(True) == sys.float_info.epsilon
    assert d.pmf(False) == 1.0 - sys.float_info.epsilon

def test_bernoulli():
    d = Lea.bernoulli(0)
    assert d.p(1) == PF(0)
    d = Lea.bernoulli(1,2)
    assert d.p(1) == PF(1,2)
    d = Lea.bernoulli(1)
    assert d.p(1) == PF(1)
    d = Lea.bernoulli(Fraction(123456789,1000000000))
    assert d.p(1) == PF(123456789,1000000000)
    d = Lea.bernoulli(Decimal("0.123456789"))
    assert d.p(1) == PF(123456789,1000000000)
    d = Lea.bernoulli("12.3456789%")
    assert d.p(1) == PF(123456789,1000000000)
    assert d.p(0) == 1 - PF(123456789,1000000000)
    d = Lea.bernoulli(0.0)
    assert d.p(1) == PF(0)
    d = Lea.bernoulli(0.5)
    assert d.p(1) == PF(1,2)
    d = Lea.bernoulli(1.0)
    assert d.p(1) == PF(1)
    # the following cases cannot assert exact values due to float representation
    # the pmf method shall convert the prob. fraction back into the given float (no loss of precision)
    d = Lea.bernoulli(0.2)
    assert d.pmf(1) == 0.2
    assert d.pmf(0) == 1.0 - 0.2
    d = Lea.bernoulli(0.123456789)
    assert d.pmf(1) == 0.123456789
    assert d.pmf(0) == 1.0 - 0.123456789
    d = Lea.bernoulli(sys.float_info.epsilon)
    assert d.pmf(1) == sys.float_info.epsilon
    assert d.pmf(0) == 1.0 - sys.float_info.epsilon

def test_poisson():
    d = Lea.poisson(2)
    # Probability of k events (mean m) is (m**k)*exp(-m)/k!
    expected = 8.0 * math.exp(-2) / 6.0
    # Result is not exact - check it is within 10 decimal places
    assert round(d.pmf(3)-expected, 10) == 0

@pytest.mark.skip(reason="Test not written yet")
def test_csv():
    Lea.fromCSVFilename
    Lea.fromCSVFile
    Lea.fromPandasDF


def test_draw_unsorted_without_replacement():
    # test an unbiased die
    d = Lea.interval(1,6)
    d0 = d.draw(0)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2)
    assert len(d2._vs) == 6*5
    assert d2.p((1,1)) == 0
    assert d2.p((6,6)) == 0
    assert d2.p((1,2)) == PF(1,6) * PF(1,5)
    assert d2.p((2,1)) == PF(1,6) * PF(1,5)
    assert d2.p((2,3)) == PF(1,6) * PF(1,5)
    d3 = d.draw(3)
    assert len(d3._vs) == 6*5*4
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 0
    assert d3.p((1,2,3)) == PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.p((2,1,3)) == PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.p((2,3,1)) == PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.p((2,3,4)) == PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.equiv(d.draw(3,sorted=False,replacement=False))
    with pytest.raises(Lea.Error):
        d7 = d.draw(7)
    # test a biased die, with P(d==1) = 2/7
    d = Lea.fromValFreqsDict({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2)
    assert len(d2._vs) == 6*5
    assert d2.p((1,1)) == 0
    assert d2.p((6,6)) == 0
    assert d2.p((1,2)) == PF(2,7) * PF(1,5)
    assert d2.p((2,1)) == PF(1,7) * PF(2,6)
    assert d2.p((2,3)) == PF(1,7) * PF(1,6)
    d3 = d.draw(3)
    assert len(d3._vs) == 6*5*4
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 0
    assert d3.p((1,2,3)) == PF(2,7) * PF(1,5) * PF(1,4)
    assert d3.p((2,1,3)) == PF(1,7) * PF(2,6) * PF(1,4)
    assert d3.p((2,3,1)) == PF(1,7) * PF(1,6) * PF(2,5)
    assert d3.p((2,3,4)) == PF(1,7) * PF(1,6) * PF(1,5)
    assert d3.equiv(d.draw(3,sorted=False,replacement=False))
    with pytest.raises(Lea.Error):
        d7 = d.draw(7)

def test_draw_unsorted_with_replacement():
    # test an unbiased die
    d = Lea.interval(1,6)
    d0 = d.draw(0,replacement=True)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1,replacement=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,replacement=True)
    assert len(d2._vs) == 6**2
    assert d2.p((1,1)) == PF(1,6) * PF(1,6)
    assert d2.p((6,6)) == PF(1,6) * PF(1,6)
    assert d2.p((1,2)) == PF(1,6) * PF(1,6)
    assert d2.p((2,1)) == PF(1,6) * PF(1,6)
    assert d2.p((2,3)) == PF(1,6) * PF(1,6)
    d3 = d.draw(3,replacement=True)
    assert len(d3._vs) == 6**3
    assert d3.p((1,2,1)) == PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((1,6,6)) == PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((1,2,3)) == PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((2,1,3)) == PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((2,3,1)) == PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((2,3,4)) == PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.equiv(d.draw(3,sorted=False,replacement=True))
    d7 = d.draw(7,replacement=True)
    assert len(d7._vs) == 6**7
    # test a biased die, with P(d==1) = 2/7
    d = Lea.fromValFreqsDict({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0,replacement=True)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1,replacement=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,replacement=True)
    assert len(d2._vs) == 6**2
    assert d2.p((1,1)) == PF(2,7) * PF(2,7)
    assert d2.p((6,6)) == PF(1,7) * PF(1,7)
    assert d2.p((1,2)) == PF(2,7) * PF(1,7)
    assert d2.p((2,1)) == PF(1,7) * PF(2,7)
    assert d2.p((2,3)) == PF(1,7) * PF(1,7)
    d3 = d.draw(3,replacement=True)
    assert len(d3._vs) == 6**3
    assert d3.p((1,2,1)) == PF(2,7) * PF(2,7) * PF(1,7)
    assert d3.p((1,6,6)) == PF(2,7) * PF(1,7) * PF(1,7)
    assert d3.p((1,2,3)) == PF(2,7) * PF(1,7) * PF(1,7)
    assert d3.p((2,1,3)) == PF(1,7) * PF(2,7) * PF(1,7)
    assert d3.p((2,3,1)) == PF(1,7) * PF(1,7) * PF(2,7)
    assert d3.p((2,3,4)) == PF(1,7) * PF(1,7) * PF(1,7)
    assert d3.equiv(d.draw(3,sorted=False,replacement=True))
    d7 = d.draw(7,replacement=True)
    assert len(d7._vs) == 6**7

def test_draw_sorted_without_replacement():
    # test an unbiased die
    d = Lea.interval(1,6)
    d0 = d.draw(0,sorted=True)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1,sorted=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True)
    assert len(d2._vs) == 15
    assert d2.p((1,1)) == 0
    assert d2.p((6,6)) == 0
    assert d2.p((1,2)) == 2 * PF(1,6) * PF(1,5)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == 2 * PF(1,6) * PF(1,5)
    assert d2.equiv(d.draw(2).map(lambda vs: tuple(sorted(vs))).getAlea())
    d3 = d.draw(3,sorted=True)
    assert len(d3._vs) == 20
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 0
    assert d3.p((1,2,3)) == 6 * PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.equiv(d.draw(3,sorted=True,replacement=False))
    assert d3.equiv(d.draw(3).map(lambda vs: tuple(sorted(vs))).getAlea())
    with pytest.raises(Lea.Error):
        d7 = d.draw(7,sorted=True)
    # test a biased die, with P(d==1) = 2/7
    d = Lea.fromValFreqsDict({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0,sorted=True)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1,sorted=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True)
    assert len(d2._vs) == 15
    assert d2.p((1,1)) == 0
    assert d2.p((6,6)) == 0
    assert d2.p((1,2)) == PF(2,7) * PF(1,5) + PF(1,7) * PF(2,6)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == PF(1,7) * PF(1,6) + PF(1,7) * PF(1,6)
    assert d2.equiv(d.draw(2).map(lambda vs: tuple(sorted(vs))).getAlea())
    d3 = d.draw(3,sorted=True)
    assert len(d3._vs) == 20
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 0
    assert d3.p((1,2,3)) == 2 * (PF(2,7) * PF(1,5) * PF(1,4) + PF(1,7) * PF(1,6) * PF(2,5) + PF(1,7) * PF(2,6) * PF(1,4))
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,7) * PF(1,6) * PF(1,5)
    assert d3.equiv(d.draw(3,sorted=True,replacement=False))
    assert d3.equiv(d.draw(3).map(lambda vs: tuple(sorted(vs))).getAlea())
    with pytest.raises(Lea.Error):
        d7 = d.draw(7,sorted=True)

def test_draw_sorted_with_replacement():
    # test an unbiased die
    d = Lea.interval(1,6)
    d0 = d.draw(0,sorted=True,replacement=True)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1,sorted=True,replacement=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True,replacement=True)
    assert len(d2._vs) == 21
    assert d2.p((1,1)) == PF(1,6) * PF(1,6)
    assert d2.p((6,6)) == PF(1,6) * PF(1,6)
    assert d2.p((1,2)) == 2 * PF(1,6) * PF(1,6)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == 2 * PF(1,6) * PF(1,6)
    assert d2.equiv(d.draw(2,replacement=True).map(lambda vs: tuple(sorted(vs))).getAlea())
    d3 = d.draw(3,sorted=True,replacement=True)
    assert len(d3._vs) == 56
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 3 * PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((1,2,3)) == 6 * PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.equiv(d.draw(3,replacement=True).map(lambda vs: tuple(sorted(vs))).getAlea())
    d7 = d.draw(7,sorted=True,replacement=True)
    assert len(d7._vs) == 792
    # test a biased die, with P(d==1) = 2/7
    d = Lea.fromValFreqsDict({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0,sorted=True,replacement=True)
    assert d0.equiv(Lea.fromVals(()))
    d1 = d.draw(1,sorted=True,replacement=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True,replacement=True)
    assert len(d2._vs) == 21
    assert d2.p((1,1)) == PF(2,7) * PF(2,7)
    assert d2.p((6,6)) == PF(1,7) * PF(1,7)
    assert d2.p((1,2)) == 2 * PF(2,7) * PF(1,7)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == 2 * PF(1,7) * PF(1,7)
    assert d2.equiv(d.draw(2,replacement=True).map(lambda vs: tuple(sorted(vs))).getAlea())
    d3 = d.draw(3,sorted=True,replacement=True)
    assert len(d3._vs) == 56
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 3 * PF(2,7) * PF(1,7) * PF(1,7)
    assert d3.p((1,2,3)) == 6 * PF(2,7) * PF(1,7) * PF(1,7)
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,7) * PF(1,7) * PF(1,7)
    assert d3.equiv(d.draw(3,replacement=True).map(lambda vs: tuple(sorted(vs))).getAlea())
    d7 = d.draw(7,sorted=True,replacement=True)
    assert len(d7._vs) == 792
