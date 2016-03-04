from lea import Lea
import pytest

def test_add():
    """Two distributions can be added together"""
    d1 = Lea.fromVals(1,2,3,4)
    d2 = Lea.fromVals(1,2,3,4)
    dsum = Lea.fromValFreqs((2,1), (3,2), (4,3), (5,4), (6,3), (7,2), (8,1))
    assert (d1 + d2).equiv(dsum)

def test_add():
    """Two distributions can be added together"""
    d1 = Lea.fromVals(1,2,3,4)
    d2 = Lea.fromVals(1,2,3,4)
    dsum = Lea.fromValFreqs((2,1), (3,2), (4,3), (5,4), (6,3), (7,2), (8,1))
    dinc = Lea.fromVals(2,3,4,5)
    assert (d1 + d2).equiv(dsum)
    assert (d1 + 1).equiv(dinc)
    assert (1 + d1).equiv(dinc)
    assert (d1).equiv(d1)

def test_sub():
    """A distribution can be subtracted from another"""
    d1 = Lea.fromVals(1,2,3,4)
    d2 = Lea.fromVals(1,2,3,4)
    ddiff = Lea.fromVals(0,-1,-2,-3,1,0,-1,-2,2,1,0,-1,3,2,1,0)
    ddec = Lea.fromVals(0,1,2,3)
    dneg = Lea.fromVals(-1,-2,-3,-4)
    assert (d1 - d2).equiv(ddiff)
    assert (d1 - 1).equiv(ddec)
    assert (1 - d1).equiv(-ddec)
    assert (-d1).equiv(dneg)

def test_mul():
    """A distribution can be multiplied by another"""
    d1 = Lea.fromVals(1,2,3,4)
    d2 = Lea.fromVals(1,2,3,4)
    dprod = Lea.fromVals(1,2,3,4,2,4,6,8,3,6,9,12,4,8,12,16)
    ddbl = Lea.fromVals(2,4,6,8)
    assert (d1 * d2).equiv(dprod)
    assert (d1 * 2).equiv(ddbl)
    assert (2 * d1).equiv(ddbl)

def test_div():
    """A distribution can be divided by another"""
    d1 = Lea.fromVals(12,24,36,48)
    d2 = Lea.fromVals(1,2,3,4)
    dquot = Lea.fromVals(12,24,36,48,6,12,18,24,4,8,12,16,3,6,9,12)
    dhalf = Lea.fromVals(6,12,18,24)
    ddiv = Lea.fromVals(12,6,4,3)
    assert (d1 / d2).equiv(dquot)
    assert (d1 / 2).equiv(dhalf)
    assert (12 / d2).equiv(ddiv)

def test_floordiv():
    """A distribution can be divided by another"""
    d1 = Lea.fromVals(12,24,36,9)
    d2 = Lea.fromVals(1,2,3,4)
    dquot = Lea.fromVals(12,24,36,9,6,12,18,4,4,8,12,3,3,6,9,2)
    dhalf = Lea.fromVals(6,12,18,4)
    ddiv = Lea.fromVals(7,3,2,1)
    assert (d1 // d2).equiv(dquot), "{}\n--------\n{}".format(d1//d2, dquot)
    assert (d1 // 2).equiv(dhalf)
    assert (7 // d2).equiv(ddiv)

def test_mod():
    """We can take mod of one distribution by another"""
    d1 = Lea.fromVals(6,7,8,9)
    d2 = Lea.fromVals(2,3)
    dmod = Lea.fromVals(0,1,0,1,0,1,2,0)
    dmod2 = Lea.fromVals(0,1)
    d12mod = Lea.fromVals(0,5,4,3)
    assert (d1 % d2).equiv(dmod)
    assert (d1 % 2).equiv(dmod2)
    assert (12 % d1).equiv(d12mod)

def test_divmod():
    """We can take divmod of one distribution by another"""
    d1 = Lea.fromVals(6,7,8,9)
    d2 = Lea.fromVals(2,3)
    ddmod = Lea.fromVals((3,0),(3,1),(4,0),(4,1),(2,0),(2,1),(2,2),(3,0))
    ddmod2 = Lea.fromVals((3,0),(3,1),(4,0),(4,1))
    d12dmod = Lea.fromVals((2,0),(1,5),(1,4),(1,3))
    assert (divmod(d1, d2)).equiv(ddmod)
    assert (divmod(d1, 2)).equiv(ddmod2)
    assert (divmod(12, d1)).equiv(d12dmod)

def test_pow():
    d1 = Lea.fromVals(1,4,9)
    d2 = Lea.fromVals(1,2,3)
    dpow = Lea.fromVals(1,4,9,1,16,81,1,64,729)
    dpow2 = Lea.fromVals(2,4,8)
    assert (d1 ** d2).equiv(dpow)
    assert (d2 ** 2).equiv(d1)
    assert (2 ** d2).equiv(dpow2)

def test_abs():
    d1 = Lea.fromVals(1,2,3,4,-1,-3,-5,0)
    dabs = Lea.fromVals(1,2,3,4,1,3,5,0)
    assert abs(d1).equiv(dabs)

def test_bool():
    with pytest.raises(Lea.Error):
        bool(Lea.fromVals(True))
    with pytest.raises(Lea.Error):
        bool(Lea.fromVals(False))
    with pytest.raises(Lea.Error):
        bool(Lea.fromVals(1,2,3))
