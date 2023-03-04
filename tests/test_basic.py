# Some basic tests as a starter

import lea
import pytest
from lea.prob_fraction import ProbFraction as PF

# All tests are made using fraction representation, in order to ease comparison
@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('r')

def test_equiprobable(setup):
    flip = lea.vals('Head','Tail')
    assert set(flip.pmf_tuple()) == {("Head", PF(1,2)), ("Tail", PF(1,2))}

def test_biased(setup):
    biasedFlip = lea.vals('Head','Tail','Tail')
    assert set(biasedFlip.pmf_tuple()) == {("Head", PF(1,3)), ("Tail", PF(2,3))}

def test_simplest_form(setup):
    flip = lea.vals("H", "H", "T", "T")
    assert set(flip.pmf_tuple()) == {("H", PF(1,2)), ("T", PF(1,2))}

def test_fromseq(setup):
    seq = lea.vals(*range(1, 7))
    vals = lea.vals(1,2,3,4,5,6)
    assert vals.equiv(seq)

def test_interval(setup):
    seq = lea.interval(1, 6)
    vals = lea.vals(1,2,3,4,5,6)
    assert vals.equiv(seq)

def test_explicit(setup):
    biased = lea.vals("H", "T", "T")
    explicit = lea.pmf((("H", PF(1,3)), ("T", PF(2,3))))
    assert biased.equiv(explicit)

def test_explicit_counts(setup):
    biased = lea.vals("H", "T", "T")
    explicit = lea.pmf((("H", 1), ("T", 2)))
    assert biased.equiv(explicit)

def test_explicitdict(setup):
    biased = lea.vals("H", "T", "T")
    explicit = lea.pmf({"H": PF(1,3), "T": PF(2,3)})
    assert biased.equiv(explicit)

def test_explicitdict_counts(setup):
    biased = lea.vals("H", "T", "T")
    explicit = lea.pmf({"H": 1, "T": 2})
    assert biased.equiv(explicit)

def test_boolprob(setup):
    explicit = lea.pmf(((True, 27), (False, 73)))
    prob = lea.event('27/100')
    assert prob.equiv(explicit)
    prob2 = lea.event(PF(27,100))
    assert prob2.equiv(explicit)

def test_bernoulli(setup):
    b = lea.bernoulli('3/10')
    explicit = lea.pmf(((0, 7), (1, 3)))
    assert b.equiv(explicit)
    b2 = lea.bernoulli(PF(3,10))
    assert b2.equiv(explicit)

def test_binom(setup):
    b = lea.binom(6, '3/10')
    p = PF(3,10)
    explicit = lea.pmf((
        (0, (1-p)**6),
        (1, 6*p*(1-p)**5),
        (2, 15*p**2*(1-p)**4),
        (3, 20*p**3*(1-p)**3),
        (4, 15*p**4*(1-p)**2),
        (5, 6*p**5*(1-p)),
        (6, p**6)
    ))
    assert b.equiv(explicit)
    b2 = lea.binom(6, p)
    assert b2.equiv(explicit)

def test_binom_bernoulli(setup):
    binom = lea.binom(1, '5/6')
    bernoulli = lea.bernoulli('5/6')
    assert binom.equiv(bernoulli)

def test_distribution_data(setup):
    # Examples from the wiki
    heights = lea.pmf(((0.5,1),(1.0,2),(1.5,4),(2.0,5),(2.5,5),(3.0,2),(3.5,1)))
    assert heights.pmf_tuple() == ((0.5, PF(1,20)), (1.0, PF(2,20)), (1.5, PF(4,20)), (2.0, PF(5,20)), (2.5, PF(5,20)), (3.0, PF(2,20)), (3.5, PF(1,20)))
    assert heights.pmf_dict() == {0.5: PF(1,20), 1.0: PF(2,20), 2.0: PF(5,20), 3.0: PF(2,20), 3.5: PF(1,20), 1.5: PF(4,20), 2.5: PF(5,20)}
    assert heights.support()== (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    assert heights.ps() == (PF(1,20), PF(2,20), PF(4,20), PF(5,20), PF(5,20), PF(2,20), PF(1,20))
    assert heights.cdf_tuple() == ((0.5, PF(1,20)), (1.0, PF(3,20)), (1.5, PF(7,20)), (2.0, PF(12,20)), (2.5, PF(17,20)), (3.0, PF(19,20)), (3.5, PF(20,20)))
    assert heights.cdf_dict() == {0.5: PF(1,20), 1.0: PF(3,20), 2.0: PF(12,20), 3.0: PF(19,20), 3.5: PF(20,20), 1.5: PF(7,20), 2.5: PF(17,20)}

def test_get_prob(setup):
    flip = lea.vals(*"HT")
    assert lea.Pf(flip == "H") == 0.5

def test_descriptive_statistics(setup):
    die = lea.vals(1,2,3,4,5,6)
    assert die.mean_f() == 3.5
    assert die.var_f() == 2.9166666666666665
    assert die.std_f() == 1.707825127659933
    assert die.cov_f(die) == 2.9166666666666665

def test_mode(setup):
    die = lea.vals(1,2,3,4,5,6)
    assert die.mode() == (1, 2, 3, 4, 5, 6)
    heights = lea.pmf(((0.5,1),(1.0,2),(1.5,4),(2.0,5),(2.5,5),(3.0,2),(3.5,1)))
    assert heights.mode() == (2.0, 2.5)

def test_entropy(setup):
    assert lea.event('1/2').entropy() == 1.0
    assert lea.vals(1,2,3,4,5,6).entropy() == 2.584962500721156
    # note accurracy difference between Python 2 and 3
    assert abs(lea.vals(1,2,3,4,5,6,6).entropy() - 2.521640636343318) < 1e-14

def test_zero_entropy(setup):
    assert lea.event(1).entropy() == 0
    assert lea.event(0).entropy() == 0
    assert lea.vals('X').entropy() == 0

def test_random_samples(setup):
    die = lea.interval(1, 6)
    # We can't test for exact answers here, as we're generating random results
    # So we look for "obvious" characteristics
    assert set(die.random(20)) <= {1, 2, 3, 4, 5, 6}

def test_random_draw(setup):
    # If we draw as many elements as there are in the set, we get all of them
    assert set(lea.interval(1, 50).random_draw(50)) == set(range(1, 51))
