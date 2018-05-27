import lea
from lea import P
from lea.toolbox import isclose
from lea.prob_fraction import ProbFraction as PF

# All tests are made using fraction representation, in order to ease comparison
lea.set_prob_type('r')

def test_arith_with_constant():
    die = lea.interval(1, 6)
    assert (die + 3).support == (4, 5, 6, 7, 8, 9)
    assert (die - 3).support == (-2, -1, 0, 1, 2, 3)
    assert (die * 2).support == (2, 4, 6, 8, 10, 12)
    assert (die / 2).support == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert (die // 2).support == (0, 1, 2, 3)
    assert (die // 3).equiv(lea.vals(0, 0, 1, 1, 1, 2))
    assert (die % 2).equiv(lea.vals(0, 1))

def test_arith_independent_vars():
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert (die1 + die2).equiv(lea.pmf({
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
    }))

def test_arith_dependent_vars():
    die = lea.interval(1, 6)
    assert (die + die).equiv(lea.vals(2, 4, 6, 8, 10, 12))

def test_arith_square_diff():
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert ((die1 - die2) ** 2).equiv(lea.pmf(((0,3), (1,5), (4,4), (9,3), (16,2), (25,1))))

def test_arith_const_result():
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert (die1 + 2*die2 - die1 - (die2 + die2)).p(0) == 1

def test_complex_expr():
    die1 = lea.interval(1, 6)
    diexN = sum(die1.new() for i in range(4))
    assert diexN.p(13) == PF(35, 324)

def test_times():
    die1 = lea.interval(1, 6)
    diexN = sum(die1.new() for i in range(4))
    assert die1.times(4).equiv(diexN)

def test_times_alt_op():
    from operator import mul
    die = lea.interval(1, 6)
    explicit = die.new() * die.new() * die.new() * die.new()
    assert die.times(4, op=mul).equiv(explicit)

def binom_bernoulli_equiv():
    binom = lea.binom(6,3,10)
    bernoulli = lea.bernoulli(3,10)
    assert binom.equiv(bernoulli.times(6))

def test_comparisons():
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

def test_logic():
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    assert P(~(die1 == 3)) == PF(5, 6)
    assert P(~(die1 == 3) | (die1 == 3)) == 1
    assert P((die1 == 4) & (die2 == 2)) == PF(1, 36)
    assert P((die1 <= 3) & (die2 > 3)) == PF(1, 4)

def test_map():
    die1 = lea.interval(1, 6)
    def parity(x):
        return "odd" if x%2==1 else "even"
    assert die1.map(parity).equiv(lea.vals("even", "odd"))
    assert die1.map(parity).map(len).equiv(lea.vals(3, 4))

def test_multiarg_map():
    die1 = lea.interval(1, 6)
    die2 = die1.new()
    dist = lea.min_of(die1,die2)
    assert dist.equiv(lea.pmf(((1,11), (2,9), (3,7), (4,5), (5,3), (6,1))))