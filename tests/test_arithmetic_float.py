import lea
import pytest
from lea import Pf
from lea.toolbox import isclose

# test made with float representation for probabilities
@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('f')

def PF(a,b):
    return float(a)/b

def test_arith_with_constant(setup):
    die = lea.interval(1, 6)
    assert (die + 3).support == (4, 5, 6, 7, 8, 9)
    assert (die - 3).support == (-2, -1, 0, 1, 2, 3)
    assert (die * 2).support == (2, 4, 6, 8, 10, 12)
    assert (die / 2).support == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert (die // 2).support == (0, 1, 2, 3)
    assert (die // 3).equiv_f(lea.vals(0, 0, 1, 1, 1, 2))
    assert (die % 2).equiv_f(lea.vals(0, 1))

def test_arith_independent_vars(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert (die1 + die2).equiv_f(lea.pmf({
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
    }))

def test_arith_dependent_vars(setup):
    die = lea.interval(1, 6)
    assert (die + die).equiv_f(lea.vals(2, 4, 6, 8, 10, 12))

def test_arith_square_diff(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert ((die1 - die2) ** 2).equiv_f(lea.pmf(((0,3), (1,5), (4,4), (9,3), (16,2), (25,1))))

def test_arith_const_result(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert (die1 + 2*die2 - die1 - (die2 + die2)).p(0) == 1

def test_complex_expr(setup):
    die1 = lea.interval(1, 6)
    diexN = sum(die1.new() for i in range(4))
    assert isclose(diexN.p(13),PF(35, 324))

def test_times(setup):
    die1 = lea.interval(1, 6)
    diexN = sum(die1.new() for i in range(4))
    assert die1.times(4).equiv_f(diexN)

def test_times_alt_op(setup):
    from operator import mul
    die = lea.interval(1, 6)
    explicit = die.new() * die.new() * die.new() * die.new()
    assert die.times(4, op=mul).equiv_f(explicit)

def binom_bernoulli_equiv(setup):
    binom = lea.binom(6,3,10)
    bernoulli = lea.bernoulli(3,10)
    assert binom.equiv_f(bernoulli.times(6))

def test_comparisons(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert isclose(Pf(die1 == 4),PF(1, 6))
    assert isclose(Pf(die1 <= 4), PF(2, 3))
    assert isclose(Pf(die1 < die2), PF(5, 12))
    assert isclose(Pf(die1 <= die2), PF(7, 12))
    assert isclose(Pf(die1 == die2), PF(1, 6))
    assert isclose(Pf(die1 == die1), 1)
    assert isclose(Pf((die1-die2)**2 == die1**2 - die2**2), PF(1, 6))
    assert isclose(Pf((die1-die2)**2 == die1**2 + die2**2), 0)
    assert isclose(Pf((die1-die2)**2 == die1**2 + die2**2 - 2*die1*die2), 1)
    assert isclose(Pf((die1+die2).is_any_of(2,3,12)), PF(1, 9))
    assert isclose(Pf((die1+die2).is_none_of(2,3,12)), PF(8, 9))

def test_logic(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert isclose(Pf(~(die1 == 3)), PF(5, 6))
    assert isclose(Pf(~(die1 == 3) | (die1 == 3)), 1)
    assert isclose(Pf((die1 == 4) & (die2 == 2)), PF(1, 36))
    assert isclose(Pf((die1 <= 3) & (die2 > 3)), PF(1, 4))

def test_map(setup):
    die1 = lea.interval(1, 6)
    def parity(x):
        return "odd" if x%2==1 else "even"
    assert die1.map(parity).equiv_f(lea.vals("even", "odd"))
    assert die1.map(parity).map(len).equiv_f(lea.vals(3, 4))

def test_multiarg_map(setup):
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    dist = lea.min_of(die1,die2)
    assert dist.equiv_f(lea.pmf(((1,11), (2,9), (3,7), (4,5), (5,3), (6,1))))
