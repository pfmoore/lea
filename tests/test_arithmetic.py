from lea import Lea, ProbFraction, P, Flea
from lea.toolbox import isclose

# TODO
def ProbFraction(a,b):
    return a/b

def test_arith_with_constant():
    die = Lea.interval(1, 6)
    assert (die + 3).vals() == (4, 5, 6, 7, 8, 9)
    assert (die - 3).vals() == (-2, -1, 0, 1, 2, 3)
    assert (die * 2).vals() == (2, 4, 6, 8, 10, 12)
    assert (die / 2).vals() == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert (die // 2).vals() == (0, 1, 2, 3)
    assert (die // 3).equivF(Lea.fromVals(0, 0, 1, 1, 1, 2))
    assert (die % 2).equivF(Lea.fromVals(0, 1))

def test_arith_independent_vars():
    die1 = Lea.interval(1, 6)
    die2 = die1.clone()
    assert (die1 + die2).equivF(Lea.fromValFreqsDict({
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
    }))

def test_arith_dependent_vars():
    die = Lea.interval(1, 6)
    assert (die + die).equivF(Lea.fromVals(2, 4, 6, 8, 10, 12))

def test_arith_square_diff():
    die1 = Lea.interval(1, 6)
    die2 = die1.clone()
    assert ((die1 - die2) ** 2).equivF(Lea.fromValFreqs((0,3), (1,5), (4,4), (9,3), (16,2), (25,1)))

def test_arith_const_result():
    die1 = Lea.interval(1, 6)
    die2 = die1.clone()
    assert (die1 + 2*die2 - die1 - (die2 + die2)).p(0) == 1

def test_complex_expr():
    die1 = Lea.interval(1, 6)
    diexN = sum(die1.clone() for i in range(4))
    #assert diexN.p(13) == ProbFraction(35, 324)
    assert isclose(diexN.p(13),ProbFraction(35, 324))

def test_times():
    die1 = Lea.interval(1, 6)
    diexN = sum(die1.clone() for i in range(4))
    assert die1.times(4).equivF(diexN)

def test_times_alt_op():
    from operator import mul
    die = Lea.interval(1, 6)
    explicit = die.clone() * die.clone() * die.clone() * die.clone()
    assert die.times(4, op=mul).equivF(explicit)

def binom_bernoulli_equiv():
    binom = Lea.binom(6,3,10)
    bernoulli = Lea.bernoulli(3,10)
    assert binom.equivF(bernoulli.times(6))

def test_comparisons():
    die1 = Lea.interval(1, 6)
    die2 = die1.clone()
    assert isclose(P(die1 == 4),ProbFraction(1, 6))
    assert isclose(P(die1 <= 4), ProbFraction(2, 3))
    assert isclose(P(die1 < die2), ProbFraction(5, 12))
    assert isclose(P(die1 <= die2), ProbFraction(7, 12))
    assert isclose(P(die1 == die2), ProbFraction(1, 6))
    assert isclose(P(die1 == die1), 1)
    assert isclose(P((die1-die2)**2 == die1**2 - die2**2), ProbFraction(1, 6))
    assert isclose(P((die1-die2)**2 == die1**2 + die2**2), 0)
    assert isclose(P((die1-die2)**2 == die1**2 + die2**2 - 2*die1*die2), 1)
    assert isclose(P((die1+die2).isAnyOf(2,3,12)), ProbFraction(1, 9))
    assert isclose(P((die1+die2).isNoneOf(2,3,12)), ProbFraction(8, 9))

def test_logic():
    die1 = Lea.interval(1, 6)
    die2 = die1.clone()
    assert isclose(P(~(die1 == 3)), ProbFraction(5, 6))
    assert isclose(P(~(die1 == 3) | (die1 == 3)), 1)
    assert isclose(P((die1 == 4) & (die2 == 2)), ProbFraction(1, 36))
    assert isclose(P((die1 <= 3) & (die2 > 3)), ProbFraction(1, 4))

def test_map():
    die1 = Lea.interval(1, 6)
    def parity(x):
        return "odd" if x%2==1 else "even"
    assert die1.map(parity).equivF(Lea.fromVals("even", "odd"))
    assert die1.map(parity).map(len).equivF(Lea.fromVals(3, 4))

def test_multiarg_map():
    die1 = Lea.interval(1, 6)
    die2 = die1.clone()
    dist = Flea.build(min,(die1,die2))
    assert dist.equivF(Lea.fromValFreqs((1,11), (2,9), (3,7), (4,5), (5,3), (6,1)))
