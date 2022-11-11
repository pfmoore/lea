# Tests for the game utility functions in leaf.py

import lea
from lea.leaf import die, dice, dice_seq, D6, flip, card_suite, card_rank, card
from lea.prob_fraction import ProbFraction as PF
import pytest

# All tests are made using fraction representation, in order to ease comparison
@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('r')

def test_constants(setup):
    """Check the constants provided behave as expected"""
    assert D6.equiv(lea.vals(1,2,3,4,5,6))
    assert flip.equiv(lea.vals(True, False))
    # Should this be cardSuit, not card_suite???
    assert card_suite.equiv(lea.vals(*"CDHS"))
    assert card_rank.equiv(lea.vals(*"A23456789TJQK"))
    cardvals = []
    for suit in "CDHS":
        for rank in "A23456789TJQK":
            cardvals.append(rank + suit)
    assert card.equiv(lea.vals(*cardvals))
    assert len(card.support) == 52

def test_dice(setup):
    """Check the basic dieroll functions"""
    assert len(die(10).support) == 10
    assert die(10).pmf_tuple == tuple((n+1,PF(1,10)) for n in range(10))
    assert len(dice(3,6).support) == 16
    assert dice(3,6).equiv(die(6) + die(6) + die(6))

def test_dice_seq_unsorted(setup):
    """Check we can get ordered sets of dice"""
    assert dice_seq(3, 6, sorted=False).equiv(lea.joint(die(6), die(6), die(6)))

def test_dice_seq_sorted(setup):
    """Check we can get unordered sets of dice"""
    dist = dice_seq(3, 6, sorted=True)
    vals = []
    for i in range(1,7):
        for j in range(i,7):
            for k in range(j,7):
                vals.append((i,j,k))
    assert tuple(dist.support) == tuple(vals)
    assert dist.p((1,1,1)) == PF(1,36)
    assert dist.p((1,2,3)) == PF(1,216)

def test_dice_seq_sorted(setup):
    """Check that the default is unordered sets of dice"""
    assert dice_seq(3, 6).equiv(dice_seq(3, 6, sorted=True))
