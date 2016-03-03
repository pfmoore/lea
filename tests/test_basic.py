# Some basic tests as a starter

from lea import Lea, Pf

def test_equiprobable():
    flip = Lea.fromVals('Head','Tail')
    assert set(flip.vps()) == {("Head", 1), ("Tail", 1)}

def test_biased():
    biasedFlip = Lea.fromVals('Head','Tail','Tail')
    assert set(biasedFlip.vps()) == {("Head", 1), ("Tail", 2)}

def test_simplest_form():
    flip = Lea.fromVals("H", "H", "T", "T")
    assert set(flip.vps()) == {("H", 1), ("T", 1)}

def test_fromseq():
    seq = Lea.fromSeq(range(1, 7))
    vals = Lea.fromVals(1,2,3,4,5,6)
    assert vals.equiv(seq)

def test_interval():
    seq = Lea.interval(1, 6)
    vals = Lea.fromVals(1,2,3,4,5,6)
    assert vals.equiv(seq)

def test_explicit():
    biased = Lea.fromVals("H", "T", "T")
    explicit = Lea.fromValFreqs(("H", 1), ("T", 2))
    assert biased.equiv(explicit)

def test_explicitdict():
    biased = Lea.fromVals("H", "T", "T")
    explicit = Lea.fromValFreqsDict({"H": 1, "T": 2})
    assert biased.equiv(explicit)

def test_boolprob():
    explicit = Lea.fromValFreqs((True, 27), (False, 73))
    prob = Lea.boolProb(27, 100)
    assert prob.equiv(explicit)

def test_bernoulli():
    b = Lea.bernoulli(3, 10)
    explicit = Lea.fromValFreqs((0, 7), (1, 3))
    assert b.equiv(explicit)

def test_binom():
    # Explicit values from the wiki
    b = Lea.binom(6, 3, 10)
    explicit = Lea.fromValFreqs(
        (0, 117649),
        (1, 302526),
        (2, 324135),
        (3, 185220),
        (4, 59535),
        (5, 10206),
        (6, 729),
    )
    assert b.equiv(explicit)

def test_binom_bernoulli():
    binom = Lea.binom(1, 5, 6)
    bernoulli = Lea.bernoulli(5, 6)
    assert binom.equiv(bernoulli)

def test_distribution_data():
    # Examples from the wiki
    heights = Lea.fromValFreqs((0.5,1),(1.0,2),(1.5,4),(2.0,5),(2.5,5),(3.0,2),(3.5,1))
    assert tuple(heights.vps()) == ((0.5, 1), (1.0, 2), (1.5, 4), (2.0, 5), (2.5, 5), (3.0, 2), (3.5, 1))
    assert dict(heights.vps()) == {0.5: 1, 1.0: 2, 2.0: 5, 3.0: 2, 3.5: 1, 1.5: 4, 2.5: 5}
    assert heights.vals() == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    assert heights.support() == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    assert heights.pmf() == (0.05, 0.1, 0.2, 0.25, 0.25, 0.1, 0.05)
    assert heights.cdf() == (0.05, 0.15, 0.35, 0.6, 0.85, 0.95, 1.0)

def test_get_prob():
    flip = Lea.fromSeq("HT")
    assert Pf(flip == "H") == 0.5

def test_descriptive_statistics():
    die = Lea.fromVals(1,2,3,4,5,6)
    assert die.mean == 3.5
    assert die.var == 2.9166666666666665
    assert die.std == 1.707825127659933

def test_mode():
    die = Lea.fromVals(1,2,3,4,5,6)
    assert die.mode == (1, 2, 3, 4, 5, 6)
    heights = Lea.fromValFreqs((0.5,1),(1.0,2),(1.5,4),(2.0,5),(2.5,5),(3.0,2),(3.5,1))
    assert heights.mode == (2.0, 2.5)

def test_entropy():
    assert Lea.boolProb(1,2).entropy == 1.0
    assert Lea.fromVals(1,2,3,4,5,6).entropy == 2.584962500721156
    assert Lea.fromVals(1,2,3,4,5,6,6).entropy == 2.521640636343318

def test_zero_entropy():
    assert Lea.boolProb(1,1).entropy == 0
    assert Lea.boolProb(0,1).entropy == 0
    assert Lea.fromVals(*(1,)).entropy == 0

def test_random_samples():
    die = Lea.interval(1, 6)
    # We can't test for exact answers here, as we're generating random results
    # So we look for "obvious" characteristics
    assert set(die.random(20)) <= {1, 2, 3, 4, 5, 6}

def test_random_draw():
    # If we draw as many elements as there are in the set, we get all of them
    assert set(Lea.interval(1, 50).randomDraw(50)) == set(range(1, 51))
