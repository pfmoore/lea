from lea.prob_fraction import ProbFraction as PF
from lea.ext_fraction import ExtFraction as EF
from lea.number import Number
from fractions import Fraction
from decimal import Decimal
import sys
import pytest

def test_pf_is_fraction():
    """Test that a ProbFraction is a Fraction"""
    pf = PF(1,2)
    assert isinstance(pf, PF)
    assert isinstance(pf, Fraction)

def test_pf_as_str():
    pf1 = PF("0")
    assert pf1 == PF(0)
    pf2 = PF("0.5")
    assert pf2 == PF(1,2)
    pf3 = PF("1")
    assert pf3 == PF(1)
    pf4 = PF("0.123456789")
    assert pf4 == PF(123456789,1000000000)
    pf5 = PF("   0.123456789000 ")
    assert pf5 == PF(123456789,1000000000)

def test_pf_from_fraction():
    """Test creating from a fraction"""
    pf = PF._from_fraction(Fraction(3, 4))
    assert pf == PF(3,4)

def test_pf_from_decimal():
    pf1 = PF(Decimal("0"))
    assert pf1 == PF(0)
    pf2 = PF(Decimal("0.5"))
    assert pf2 == PF(1,2)
    pf3 = PF(Decimal("1"))
    assert pf3 == PF(1)
    pf4 = PF(Decimal("0.123456789"))
    assert pf4 == PF(123456789,1000000000)
    pf5 = PF(Decimal("   0.123456789000 "))
    assert pf5 == PF(123456789,1000000000)

def test_pf_from_float():
    """Test creating from a float"""
    # Checking float numbers that can be represented exactly as fractions,
    # i.e. zero and multiples of powers of 2
    pf1 = PF(0.0)
    assert pf1 == PF(0)
    pf2 = PF(1.0)
    assert pf2 == PF(1)
    pf3 = PF(0.5)
    assert pf3 == PF(1,2)
    pf4 = PF(3.0/4.0)
    assert pf4 == PF(3,4)
    # Checking float numbers that cannot be represented exactly as fractions
    # e.g. PF(0.2) is not exactly the same as fraction PF(2,10)
    # we expect no loss of accuracy when converting back the fraction to float
    pf5 = PF(0.2)
    assert float(pf5) == 0.2
    pf6 = PF(2.0/3.0)
    assert float(pf6) == 2.0/3.0
    pf7 = PF(0.123456789)
    assert float(pf7) == 0.123456789
    pf8 = PF(sys.float_info.epsilon)
    assert float(pf8) ==  sys.float_info.epsilon
    pf9 = PF(1.0-sys.float_info.epsilon)
    assert float(pf9) ==  1.0-sys.float_info.epsilon

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
    assert pf1+pf2 == EF(5,6)
    assert isinstance(pf1+pf2, EF)
    assert pf1-pf2 == EF(1,6)
    assert isinstance(pf1-pf2, EF)
    assert pf1*pf2 == EF(1,6)
    assert isinstance(pf1*pf2,EF)
    assert pf1/pf2 == EF(3,2)
    assert isinstance(pf1/pf2, EF)
    # Check reversed operators (__radd__ etc)
    assert f+pf2 == EF(5,6)
    assert isinstance(f+pf2, EF)
    assert f-pf2 == EF(1,6)
    assert isinstance(f-pf2, EF)
    assert f*pf2 == EF(1,6)
    assert isinstance(f*pf2,EF)
    assert f/pf2 == EF(3,2)
    assert isinstance(f/pf2, EF)
    # Other operators
    assert +pf1==EF(1,2)
    assert isinstance(+pf1,EF)
    assert -pf1==EF(-1,2)
    assert isinstance(-pf1, EF)
    assert pf1**2==EF(1,4)
    assert isinstance(pf1**2, EF)

def test_check():
    """Test the check method for confirming a fraction is in [0,1]"""
    PF(1,2).check()
    PF(1,1).check()
    PF(0,1).check()
    with pytest.raises(Number.Error):
        PF(2,1).check()
    with pytest.raises(Number.Error):
        PF(-1,1).check()

def test_float_str():
    """Check we can get a float representation of a ProbFraction"""
    assert isinstance(PF(1,2).as_float(), str)
    assert PF(1,2).as_float() == "0.5"

def test_pct_str():
    """Check we can get a percentage representation of a ProbFraction"""
    pct = PF(1,2).as_pct()
    assert isinstance(pct, str)
    # Don't check the exact representation, just that it looks right
    assert pct.endswith('%')
    assert float(pct[:-1]) == 50

def test_string_rep():
    """Test that string representation works"""
    # Depends on the string representation of Fraction
    assert str(PF(1,2)) == str(Fraction(1,2))
    # Fractions outside [0,1] raise an error
    with pytest.raises(Number.Error):
        str(PF(3,2))

