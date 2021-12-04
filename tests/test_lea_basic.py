import lea
from lea.prob_fraction import ProbFraction as PF
from fractions import Fraction
from decimal import Decimal
import pytest
import math
import sys

# All tests are made using fraction representation, in order to ease comparison
@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('r')

def test_new(setup):
    dist1 = lea.vals(1,2,3,4)
    dist2 = dist1.new()
    assert dist1.equiv(dist2)
    assert dist1 is not dist2

def test_id(setup):
    dist1 = lea.vals(1,2,3,4)
    dist2 = dist1.new()
    assert dist1._id() != dist2._id()
    assert isinstance(dist1._id(), str)

def test_get_leaves_set(setup):
    dist1 = lea.vals(1,2,3,4)
    dist2 = lea.vals(2,4,6,8)
    distcalc1 = (dist1 + dist2) * (dist1 - dist2)
    assert distcalc1.get_leaves_set() == {dist1, dist2}
    dist3 = lea.binom(4,0.2) 
    dist4 = lea.poisson(7)
    distcalc2 = dist3 + dist4 
    assert distcalc2.get_leaves_set() == {dist3, dist4}
    distcalc3 = distcalc1 - distcalc2 
    assert distcalc3.get_leaves_set() == {dist1, dist2, dist3, dist4}

def test_is_dependent_of(setup):
    dist1 = lea.vals(1,2,3,4)
    dist2 = lea.vals(2,4,6,8)
    distcalc1 = (dist1 + dist2) * (dist1 - dist2)
    assert dist1.is_dependent_of(dist1)
    assert distcalc1.is_dependent_of(distcalc1)
    assert distcalc1.is_dependent_of(dist1)
    assert distcalc1.is_dependent_of(dist2)
    assert dist1.is_dependent_of(distcalc1)
    assert dist2.is_dependent_of(distcalc1)
    assert dist1.is_dependent_of(dist2.given(distcalc1==0))
    assert dist1.given(distcalc1==0).is_dependent_of(dist2)
    assert not dist1.is_dependent_of(dist2)
    assert not dist2.is_dependent_of(dist1)
    dist3 = lea.binom(4,0.2)
    dist4 = lea.poisson(7)
    distcalc2 = dist3 + dist4 
    assert dist3.is_dependent_of(dist3)
    assert distcalc2.is_dependent_of(dist3)
    assert distcalc2.is_dependent_of(dist3)
    assert dist3.is_dependent_of(distcalc2)
    assert dist4.is_dependent_of(distcalc2)
    assert dist3.is_dependent_of(dist2.given(distcalc2==0))
    assert dist3.given(distcalc2==0).is_dependent_of(dist4)
    assert not distcalc2.is_dependent_of(distcalc1)
    assert not distcalc1.is_dependent_of(distcalc2)
    assert not dist3.is_dependent_of(dist4)
    assert not dist4.is_dependent_of(dist3)
    assert not dist3.is_dependent_of(distcalc1)
    assert not dist4.is_dependent_of(dist1)
    assert not distcalc1.is_dependent_of(dist4)
    assert not distcalc1.is_dependent_of(dist3)
    distcalc3 = distcalc1 - distcalc2 
    assert distcalc3.is_dependent_of(distcalc3)
    assert distcalc3.is_dependent_of(distcalc1)
    assert distcalc3.is_dependent_of(dist1)
    assert distcalc1.is_dependent_of(distcalc3)
    assert dist3.is_dependent_of(distcalc3)
    assert dist4.is_dependent_of(distcalc3)
    assert dist3.is_dependent_of(dist2.given(distcalc3==0))
    assert dist3.given(distcalc3==0).is_dependent_of(dist4)
    assert dist1.given(distcalc3==0).is_dependent_of(dist4)

# Constructors
def test_fromvals(setup):
    d = lea.vals(1,2, prob_type='f', ordered=True, sorting=False, normalization=False, check=False)

def test_fromvals_errors(setup):
    # Must be at least one value
    with pytest.raises(lea.Lea.Error):
        d = lea.vals()
    # No invalid keyword args
    with pytest.raises(lea.Lea.Error):
        d = lea.vals(1,2, foo=3)
    # Cannot have both ordered and sorting
    with pytest.raises(lea.Lea.Error):
        d = lea.vals(1,2, ordered=True, sorting=True)
    # Cannot have duplicates with ordered
    with pytest.raises(lea.Lea.Error):
        d = lea.vals(1,1, ordered=True)
    # Cannot have ordered with dictionary
    with pytest.raises(lea.Lea.Error):
        d = lea.pmf({1: 2, 2: 5}, ordered=True)

def test_fromvals_ordered(setup):
    d = lea.vals(2,1,3, ordered=True)
    assert d.support == (2,1,3)
    d = lea.pmf(((2,9),(1,7),(3,5)), ordered=True)
    assert d.support == (2,1,3)

def test_fromvals_sorting(setup):
    d = lea.vals(2,1,3,2, sorting=True)
    assert d.support == (1,2,3)

def test_from_dict(setup):
    d = lea.pmf({'a': 5, 'b': 6})
    assert set(d.pmf_tuple) == {('a', PF(5,11)), ('b', PF(6,11))}

def test_event(setup):
    d = lea.event(0)
    assert d.p(True) == PF(0)
    d = lea.event('1/2')
    assert d.p(True) == PF(1,2)
    d = lea.event(1)
    assert d.p(True) == PF(1)
    d = lea.event(Fraction(123456789,1000000000))
    assert d.p(True) == PF(123456789,1000000000)
    d = lea.event(Decimal("0.123456789"))
    assert d.p(True) == PF(123456789,1000000000)
    d = lea.event(0.0)
    assert d.p(True) == PF(0)
    d = lea.event(0.5)
    assert d.p(True) == PF(1,2)
    d = lea.event(1.0)
    assert d.p(True) == PF(1)
    # the following cases cannot assert exact values due to float representation
    # the pmf method shall convert the prob. fraction back into the given float (no loss of precision)
    d = lea.event(0.2, prob_type='f')
    assert d.p(True) == 0.2
    assert d.p(False) == 1.0 - 0.2
    d = lea.event(0.123456789, prob_type='f')
    assert d.p(True) == 0.123456789
    assert d.p(False) == 1.0 - 0.123456789
    d = lea.event(sys.float_info.epsilon, prob_type='f')
    assert d.p(True) == sys.float_info.epsilon
    assert d.p(False) == 1.0 - sys.float_info.epsilon

def test_bernoulli(setup):
    d = lea.bernoulli(0)
    assert d.p(1) == PF(0)
    d = lea.bernoulli('1/2')
    assert d.p(1) == PF(1,2)
    d = lea.bernoulli(1)
    assert d.p(1) == PF(1)
    d = lea.bernoulli(Fraction(123456789,1000000000))
    assert d.p(1) == PF(123456789,1000000000)
    d = lea.bernoulli(Decimal("0.123456789"))
    assert d.p(1) == PF(123456789,1000000000)
    d = lea.bernoulli(0.0)
    assert d.p(1) == PF(0)
    d = lea.bernoulli(0.5)
    assert d.p(1) == PF(1,2)
    d = lea.bernoulli(1.0)
    assert d.p(1) == PF(1)
    # the following cases cannot assert exact values due to float representation
    # the pmf method shall convert the prob. fraction back into the given float (no loss of precision)
    d = lea.bernoulli(0.2, prob_type='f')
    assert d.p(1) == 0.2
    assert d.p(0) == 1.0 - 0.2
    d = lea.bernoulli(0.123456789, prob_type='f')
    assert d.p(1) == 0.123456789
    assert d.p(0) == 1.0 - 0.123456789
    d = lea.bernoulli(sys.float_info.epsilon, prob_type='f')
    assert d.p(1) == sys.float_info.epsilon
    assert d.p(0) == 1.0 - sys.float_info.epsilon

def test_poisson(setup):
    d = lea.poisson(2)
    # Probability of k events (mean m) is (m**k)*exp(-m)/k!
    expected = 8.0 * math.exp(-2) / 6.0
    # Result is not exact - check it is within 10 decimal places
    assert round(d.p(3)-expected, 10) == 0

@pytest.mark.skip(reason="Test not written yet")
def test_csv(setup):
    lea.read_csv_file
    lea.read_pandas_df

def test_draw_unsorted_without_replacement(setup):
    # test an unbiased die
    d = lea.interval(1,6)
    d0 = d.draw(0)
    assert d0.equiv(lea.vals(()))
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
    with pytest.raises(lea.Lea.Error):
        d7 = d.draw(7)
    # test a biased die, with P(d==1) = 2/7
    d = lea.pmf({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0)
    assert d0.equiv(lea.vals(()))
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
    with pytest.raises(lea.Lea.Error):
        d7 = d.draw(7)

# TODO LOOP
def test_draw_unsorted_with_replacement(setup):
    # test an unbiased die
    d = lea.interval(1,6)
    d0 = d.draw(0,replacement=True)
    assert d0.equiv(lea.vals(()))
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
    d4 = d.draw(4,replacement=True)
    assert len(d4._vs) == 6**4
    # test a biased die, with P(d==1) = 2/7
    d = lea.pmf({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0,replacement=True)
    assert d0.equiv(lea.vals(()))
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
    d4 = d.draw(4,replacement=True)
    assert len(d4._vs) == 6**4

def test_draw_sorted_without_replacement(setup):
    # test an unbiased die
    d = lea.interval(1,6)
    d0 = d.draw(0,sorted=True)
    assert d0.equiv(lea.vals(()))
    d1 = d.draw(1,sorted=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True)
    assert len(d2._vs) == 15
    assert d2.p((1,1)) == 0
    assert d2.p((6,6)) == 0
    assert d2.p((1,2)) == 2 * PF(1,6) * PF(1,5)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == 2 * PF(1,6) * PF(1,5)
    assert d2.equiv(d.draw(2).map(lambda vs: tuple(sorted(vs))).get_alea())
    d3 = d.draw(3,sorted=True)
    assert len(d3._vs) == 20
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 0
    assert d3.p((1,2,3)) == 6 * PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,6) * PF(1,5) * PF(1,4)
    assert d3.equiv(d.draw(3,sorted=True,replacement=False))
    assert d3.equiv(d.draw(3).map(lambda vs: tuple(sorted(vs))).get_alea())
    with pytest.raises(lea.Lea.Error):
        d7 = d.draw(7,sorted=True)
    # test a biased die, with P(d==1) = 2/7
    d = lea.pmf({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0,sorted=True)
    assert d0.equiv(lea.vals(()))
    d1 = d.draw(1,sorted=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True)
    assert len(d2._vs) == 15
    assert d2.p((1,1)) == 0
    assert d2.p((6,6)) == 0
    assert d2.p((1,2)) == PF(2,7) * PF(1,5) + PF(1,7) * PF(2,6)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == PF(1,7) * PF(1,6) + PF(1,7) * PF(1,6)
    assert d2.equiv(d.draw(2).map(lambda vs: tuple(sorted(vs))).get_alea())
    d3 = d.draw(3,sorted=True)
    assert len(d3._vs) == 20
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 0
    assert d3.p((1,2,3)) == 2 * (PF(2,7) * PF(1,5) * PF(1,4) + PF(1,7) * PF(1,6) * PF(2,5) + PF(1,7) * PF(2,6) * PF(1,4))
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,7) * PF(1,6) * PF(1,5)
    assert d3.equiv(d.draw(3,sorted=True,replacement=False))
    assert d3.equiv(d.draw(3).map(lambda vs: tuple(sorted(vs))).get_alea())
    with pytest.raises(lea.Lea.Error):
        d7 = d.draw(7,sorted=True)

def test_draw_sorted_with_replacement(setup):
    # test an unbiased die
    d = lea.interval(1,6)
    d0 = d.draw(0,sorted=True,replacement=True)
    assert d0.equiv(lea.vals(()))
    d1 = d.draw(1,sorted=True,replacement=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True,replacement=True)
    assert len(d2._vs) == 21
    assert d2.p((1,1)) == PF(1,6) * PF(1,6)
    assert d2.p((6,6)) == PF(1,6) * PF(1,6)
    assert d2.p((1,2)) == 2 * PF(1,6) * PF(1,6)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == 2 * PF(1,6) * PF(1,6)
    assert d2.equiv(d.draw(2,replacement=True).map(lambda vs: tuple(sorted(vs))).get_alea())
    d3 = d.draw(3,sorted=True,replacement=True)
    assert len(d3._vs) == 56
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 3 * PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((1,2,3)) == 6 * PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,6) * PF(1,6) * PF(1,6)
    assert d3.equiv(d.draw(3,replacement=True).map(lambda vs: tuple(sorted(vs))).get_alea())
    d7 = d.draw(7,sorted=True,replacement=True)
    assert len(d7._vs) == 792
    # test a biased die, with P(d==1) = 2/7
    d = lea.pmf({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    d0 = d.draw(0,sorted=True,replacement=True)
    assert d0.equiv(lea.vals(()))
    d1 = d.draw(1,sorted=True,replacement=True)
    assert d1.equiv(d.map(lambda v: (v,)))
    d2 = d.draw(2,sorted=True,replacement=True)
    assert len(d2._vs) == 21
    assert d2.p((1,1)) == PF(2,7) * PF(2,7)
    assert d2.p((6,6)) == PF(1,7) * PF(1,7)
    assert d2.p((1,2)) == 2 * PF(2,7) * PF(1,7)
    assert d2.p((2,1)) == 0
    assert d2.p((2,3)) == 2 * PF(1,7) * PF(1,7)
    assert d2.equiv(d.draw(2,replacement=True).map(lambda vs: tuple(sorted(vs))).get_alea())
    d3 = d.draw(3,sorted=True,replacement=True)
    assert len(d3._vs) == 56
    assert d3.p((1,2,1)) == 0
    assert d3.p((1,6,6)) == 3 * PF(2,7) * PF(1,7) * PF(1,7)
    assert d3.p((1,2,3)) == 6 * PF(2,7) * PF(1,7) * PF(1,7)
    assert d3.p((2,1,3)) == 0
    assert d3.p((2,3,1)) == 0
    assert d3.p((2,3,4)) == 6 * PF(1,7) * PF(1,7) * PF(1,7)
    assert d3.equiv(d.draw(3,replacement=True).map(lambda vs: tuple(sorted(vs))).get_alea())
    d7 = d.draw(7,sorted=True,replacement=True)
    assert len(d7._vs) == 792
