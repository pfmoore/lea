# Tests for the game utility functions in leaf.py

from lea import die, dice, diceSeq, D6, flip, cardSuite, cardRank, card
from lea import Lea
from lea import ProbFraction as PF

def test_constants():
    """Check the constants provided behave as expected"""
    assert D6.equiv(Lea.fromSeq([1,2,3,4,5,6]))
    assert flip.equiv(Lea.fromSeq([True, False]))
    # Should this be cardSuit, not cardSuite???
    assert cardSuite.equiv(Lea.fromSeq("CDHS"))
    assert cardRank.equiv(Lea.fromSeq("A23456789TJQK"))
    cardvals = []
    for suit in "CDHS":
        for rank in "A23456789TJQK":
            cardvals.append(rank + suit)
    assert card.equiv(Lea.fromSeq(cardvals))
    assert len(card.vals()) == 52

def test_dice():
    """Check the basic dieroll functions"""
    assert len(die(10).vals()) == 10
    assert tuple(die(10).vps()) == tuple((n+1,1) for n in range(10))
    assert len(dice(3,6).vals()) == 16
    assert dice(3,6).equiv(die(6) + die(6) + die(6))

def test_dice_seq_unsorted():
    """Check we can get ordered sets of dice"""
    assert diceSeq(3, 6, sorted=False).equiv(Lea.cprod(die(6), die(6), die(6)))

def test_dice_seq_sorted():
    """Check we can get unordered sets of dice"""
    dist = diceSeq(3, 6, sorted=True)
    vals = []
    for i in range(1,7):
        for j in range(i,7):
            for k in range(j,7):
                vals.append((i,j,k))
    assert tuple(dist.vals()) == tuple(vals)
    assert dist.p((1,1,1)) == PF(1,36)
    assert dist.p((1,2,3)) == PF(1,216)

def test_dice_seq_sorted():
    """Check that the default is unordered sets of dice"""
    assert diceSeq(3, 6).equiv(diceSeq(3, 6, sorted=True))
