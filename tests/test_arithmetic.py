import lea
import pytest
from lea import P
from lea.toolbox import isclose
from lea.prob_fraction import ProbFraction as PF
from lea.ext_fraction import ExtFraction as EF

# All tests are made using fraction representation, in order to ease comparison
@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('r')
    
def test_arith_with_constant(setup):
    die = lea.interval(1, 6)
    assert (die + 3).support()== (4, 5, 6, 7, 8, 9)
    assert (die - 3).support()== (-2, -1, 0, 1, 2, 3)
    assert (die * 2).support()== (2, 4, 6, 8, 10, 12)
    assert (die / 2).support()== (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert (die // 2).support()== (0, 1, 2, 3)
    assert (die // 3).equiv(lea.vals(0, 0, 1, 1, 1, 2))
    assert (die % 2).equiv(lea.vals(0, 1))

def test_arith_independent_vars(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert (die1 + die2).equiv(lea.pmf({
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
    }))

def test_arith_dependent_vars(setup):
    die = lea.interval(1, 6)
    assert (die + die).equiv(lea.vals(2, 4, 6, 8, 10, 12))

def test_arith_square_diff(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert ((die1 - die2) ** 2).equiv(lea.pmf(((0,3), (1,5), (4,4), (9,3), (16,2), (25,1))))

def test_arith_const_result(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert (die1 + 2*die2 - die1 - (die2 + die2)).p(0) == 1

def test_complex_expr(setup):
    die1 = lea.interval(1, 6)
    diexN = sum(die1.new() for i in range(4))
    assert diexN.p(13) == PF(35, 324)

def test_times(setup):
    die1 = lea.interval(1, 6)
    diexN = sum(die1.new() for i in range(4))
    assert die1.times(4).equiv(diexN)

def test_given_times(setup):
    die1 = lea.interval(1, 6)    
    assert die1.given(die1<=2).times(2).equiv(lea.pmf(((2,1), (3,2), (4,1))))

def test_times_alt_op(setup):
    from operator import mul
    die = lea.interval(1, 6)
    explicit = die.new() * die.new() * die.new() * die.new()
    assert die.times(4, op=mul).equiv(explicit)

def test_bernoulli_corner_cases(setup):
    assert lea.bernoulli(PF(0)).equiv(0)
    assert lea.bernoulli(PF(1)).equiv(1)
    assert lea.bernoulli(0.0).equiv(0)
    assert lea.bernoulli(1.0).equiv(1)

def test_binom_corner_cases(setup):
    assert lea.binom(0,PF(0)).equiv(0)
    assert lea.binom(0,PF(1)).equiv(0)
    assert lea.binom(6,PF(0)).equiv(0)
    assert lea.binom(6,PF(1)).equiv(6)
    assert lea.binom(0,0.0).equiv(0)
    assert lea.binom(0,1.0).equiv(0)
    assert lea.binom(6,0.0).equiv(0)
    assert lea.binom(6,1.0).equiv(6)

def test_binom_bernoulli_equiv(setup):
    binom = lea.binom(6,PF(3,10))
    bernoulli = lea.bernoulli(PF(3,10))
    assert binom.equiv(bernoulli.times(6))

def test_comparisons(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert P(die1 == 4) == PF(1, 6)
    assert P(die1 <= 4) == PF(2, 3)
    assert P(die1 < die2) == PF(5, 12)
    assert P(die1 <= die2) == PF(7, 12)
    assert P(die1 == die2) == PF(1, 6)
    assert P(die1 == die1) == 1
    assert P((die1-die2)**2 == die1**2 - die2**2) == PF(1, 6)
    assert P((die1-die2)**2 == die1**2 + die2**2) == 0
    assert P((die1-die2)**2 == die1**2 + die2**2 - 2*die1*die2) == 1
    assert P((die1+die2).is_any_of(2,3,12)) == PF(1, 9)
    assert P((die1+die2).is_none_of(2,3,12)) == PF(8, 9)

def test_logic(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert P(~(die1 == 3)) == PF(5, 6)
    assert P(~(die1 == 3) | (die1 == 3)) == 1
    assert P((die1 == 4) & (die2 == 2)) == PF(1, 36)
    assert P((die1 <= 3) & (die2 > 3)) == PF(1, 4)

def test_map(setup):
    die1 = lea.interval(1, 6)
    def parity(x):
        return "odd" if x%2==1 else "even"
    assert die1.map(parity).equiv(lea.vals("even", "odd"))
    assert die1.map(parity).map(len).equiv(lea.vals(3, 4))

def test_min_of(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    die3 = die1.new()
    assert lea.min_of(die1).equiv(die1)
    assert lea.min_of(die1,die2).equiv(lea.pmf({1: 11, 2: 9, 3: 7, 4: 5, 5: 3, 6: 1}))
    assert lea.min_of(die1,die2,die3).equiv(lea.pmf({1: 91, 2: 61, 3: 37, 4: 19, 5: 7, 6: 1}))
    assert lea.min_of(die1,fast=True).equiv(die1)
    assert lea.min_of(die1,die2,fast=True).equiv(lea.pmf({1: 11, 2: 9, 3: 7, 4: 5, 5: 3, 6: 1}))
    assert lea.min_of(die1,die2,die3,fast=True).equiv(lea.pmf({1: 91, 2: 61, 3: 37, 4: 19, 5: 7, 6: 1}))
        
def test_max_of(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    die3 = die1.new()
    assert lea.max_of(die1).equiv(die1)
    assert lea.max_of(die1,die2).equiv(lea.pmf({1: 1, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11}))
    assert lea.max_of(die1,die2,die3).equiv(lea.pmf({1: 1, 2: 7, 3: 19, 4: 37, 5: 61, 6: 91}))
    assert lea.max_of(die1,fast=True).equiv(die1)
    assert lea.max_of(die1,die2,fast=True).equiv(lea.pmf({1: 1, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11}))
    assert lea.max_of(die1,die2,die3,fast=True).equiv(lea.pmf({1: 1, 2: 7, 3: 19, 4: 37, 5: 61, 6: 91}))

def test_covariance_1(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    dice = die1 + die2
    assert die1.cov(die2) == 0
    assert die1.var() == EF(35,12)
    assert die1.cov(die1) == EF(35,12)
    assert dice.cov(die1) == EF(35,12)
    assert die1.cov(dice) == EF(35,12)

def test_covariance_2(setup):
    # see http://en.wikipedia.org/wiki/Covariance
    a = lea.pmf({(8,5): '0/10 ', (8,6): '4/10', (8,7): '1/10',
                 (9,5): '3/10 ', (9,6): '0/10', (9,7): '2/10'})
    y = a[0]
    x = a[1]
    assert x.cov(y) == EF(-1,10)
    assert y.cov(x) == EF(-1,10)
