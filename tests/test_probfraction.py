from lea import ProbFraction as PF
from fractions import Fraction
import pytest

def test_pf_is_fraction():
    """Test that a ProbFraction is a Fraction"""
    pf = PF(1,2)
    assert isinstance(pf, PF)
    assert isinstance(pf, Fraction)

def test_pf_as_pct():
    """Test creating from a percentage"""
    pf = PF("50%")
    assert pf == PF(1,2)

def test_pf_from_fraction():
    """Test creating from a fraction"""
    pf = PF.fromFraction(Fraction(3,4))
    assert pf == PF(3,4)

def test_coerce():
    """Test coercing values to ProbFractions"""
    pf1 = PF(5,6)
    pf2 = PF.coerce(pf1)
    # Coercing a PF does nothing
    assert pf1 is pf2
    f = Fraction(7,8)
    # Coercing a fraction makes it a PF
    pf3 = PF.coerce(f)
    assert isinstance(pf3, PF)
    assert pf3 == f

def test_arithmetic():
    """Check arithmetic operators for correctness (result and type)"""
    pf1 = PF(1,2)
    pf2 = PF(1,3)
    f = Fraction(1,2)
    # Basic arithmetic
    assert pf1+pf2 == PF(5,6)
    assert isinstance(pf1+pf2, PF)
    assert pf1-pf2 == PF(1,6)
    assert isinstance(pf1-pf2, PF)
    assert pf1*pf2 == PF(1,6)
    assert isinstance(pf1*pf2, PF)
    assert pf1/pf2 == PF(3,2)
    assert isinstance(pf1/pf2, PF)
    # Check reversed operators (__radd__ etc)
    assert f+pf2 == PF(5,6)
    assert isinstance(f+pf2, PF)
    assert f-pf2 == PF(1,6)
    assert isinstance(f-pf2, PF)
    assert f*pf2 == PF(1,6)
    assert isinstance(f*pf2, PF)
    assert f/pf2 == PF(3,2)
    assert isinstance(f/pf2, PF)
    # Other operators
    assert +pf1==PF(1,2)
    assert isinstance(+pf1, PF)
    assert -pf1==PF(-1,2)
    assert isinstance(-pf1, PF)
    assert pf1**2==PF(1,4)
    assert isinstance(pf1**2, PF)

def test_check():
    """Test the check method for confirming a fraction is in [0,1]"""
    PF(1,2).check()
    PF(1,1).check()
    PF(0,1).check()
    with pytest.raises(PF.Error):
        PF(2,1).check()
    with pytest.raises(PF.Error):
        PF(-1,1).check()

def test_float_str():
    """Check we can get a float representation of a ProbFraction"""
    assert isinstance(PF(1,2).asFloat(), str)
    assert PF(1,2).asFloat() == "0.5"

def test_pct_str():
    """Check we can get a percentage representation of a ProbFraction"""
    pct = PF(1,2).asPct()
    assert isinstance(pct, str)
    # Don't check the exact representation, just that it looks right
    assert pct.endswith('%')
    assert float(pct[:-1]) == 50

def test_string_rep():
    """Test that string representation works"""
    # Depends on the string representation of Fraction
    assert str(PF(1,2)) == str(Fraction(1,2))
    # Fractions outside [0,1] raise an error
    with pytest.raises(PF.Error):
        str(PF(3,2))

def test_prob_weights():
    """Check calculation of probability weights"""
    weights = PF.getProbWeights([PF(1,2), PF(1,3), PF(1,6)])
    assert weights == (3, 2, 1)
    # Trying to get the weights of a 0-length list raises an error
    with pytest.raises(PF.Error):
        PF.getProbWeights([])
