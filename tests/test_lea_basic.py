from lea import Lea
from lea import ProbFraction as PF
import pytest
import math

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
    d = Lea.boolProb(1,2)
    assert d.p(True) == PF(1,2)
    d = Lea.boolProb(PF(1,2))
    assert d.p(True) == PF(1,2)

def test_bernoulli():
    d = Lea.bernoulli(1,2)
    assert d.p(1) == PF(1,2)
    d = Lea.bernoulli(PF(1,2))
    assert d.p(1) == PF(1,2)

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
