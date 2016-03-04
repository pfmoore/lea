from lea import Lea, D6
from lea import ProbFraction as PF
import pytest

def test_times_commutativity():
    """See issue #3 on bitbucket"""
    lr = D6.times(2) + D6
    rl = D6 + D6.times(2)
    assert lr.equiv(rl)

def test_withprob():
    """See issues #8 and #15 on Bitbucket"""
    dU = D6.withProb(D6>=5,1,2)
    assert dU.p(5) == PF(2,8)
    assert dU.equiv(Lea.fromValFreqs((1,1),(2,1),(3,1),(4,1),(5,2),(6,2)))

def test_infeasible_cpt():
    """See issue #13 on Bitbucket"""
    a = Lea.fromSeq("xy")
    always_s = Lea.buildCPT((a == a, 's'), (None, 't'))
    never_s = Lea.buildCPT((a != a, 's'), (None, 't'))
    assert always_s.equiv('s')
    assert never_s.equiv('t')

def test_cpt_refactoring():
    """See issues #9 and #11 on Bitbucket"""
    x = Lea.boolProb(1,3)
    y = Lea.boolProb(1,4)
    z1 = Lea.buildCPT(
            (~x & ~y, Lea.boolProb(1,5)),
            (~x &  y, Lea.boolProb(1,7)),
            ( x & ~y, Lea.boolProb(1,2)),
            ( x &  y, Lea.boolProb(1,2)),
        )
    z2 = Lea.buildCPT(
            (~x & ~y, Lea.boolProb(1,5)),
            (~x &  y, Lea.boolProb(1,7)),
            ( x     , Lea.boolProb(1,2)),
        )
    z0 = Lea.buildCPT(
            (~y, Lea.boolProb(1,5)),
            ( y, Lea.boolProb(1,7)),
        )
    z3 = Lea.buildCPT(
            (~x, z0),
            ( x, Lea.boolProb(1,2)),
        )
    assert z2.equiv(z1)
    assert z3.equiv(z1)

def test_draw_nonuniform():
    """See issue #19 on bitbucket"""
    h = Lea.fromValFreqs(("A",3),("B",2),("C",1))
    expected = Lea.fromValFreqs(
            (('A', 'B'), 20),
            (('A', 'C'), 10),
            (('B', 'A'), 15),
            (('B', 'C'),  5),
            (('C', 'A'),  6),
            (('C', 'B'),  4),
        )
    assert h.draw(2).equiv(expected)

def test_joint():
    """See issue #20 on bitbucket"""
    joint = Lea.fromValFreqs(((1,2),10), ((1,3),9), ((2,2),8)).asJoint('A','B')
    assert joint.p((1,2)) == PF(10,27)
    assert joint.pmf((1,2)) == 10.0 / 27.0

def test_check_bool():
    """See issue #24 on bitbucket"""
    mixed = Lea.fromVals(True,False,12)
    with pytest.raises(Lea.Error):
        mixed.isTrue()
    with pytest.raises(Lea.Error):
        mixed.isFeasible()
    with pytest.raises(Lea.Error):
        mixed.P
    with pytest.raises(Lea.Error):
        mixed.Pf

def test_given_times():
    """See issue #28 on bitbucket"""
    flip = Lea.fromVals(0,1)
    flip4 = flip.times(4)
    expected = Lea.fromValFreqs((0,1), (1,4), (2,6))
    assert flip4.given(flip4<=2).equiv(expected)
