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
    assert (z1.given(y)).equiv(Lea.boolProb(11,42))
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
    assert (z2.given(y)).equiv(z1.given(y))
    assert (z3.given(y)).equiv(z1.given(y))

def test_cpt_refactoring_2():
    """See issues #9 and #11 on Bitbucket"""
    x = Lea.boolProb(1,3)
    y = Lea.boolProb(1,4)
    z1 = Lea.cprod(x,y).switch({
            ( True, False) : Lea.boolProb(1,2),
            ( True, True ) : Lea.boolProb(1,2),
            (False, False) : Lea.boolProb(1,5),
            (False, True ) : Lea.boolProb(1,7)}
        )
    #assert (z1.given(y)).equiv(Lea.boolProb(11,42))
    z0 = y.switch({
            False : Lea.boolProb(1,5),
            True  : Lea.boolProb(1,7) }
        )
    z3 = x.switch({
            False : z0,
            True  : Lea.boolProb(1,2) }
        )
    assert z3.equiv(z1)
    assert (z3.given(y)).equiv(z1.given(y))

    z4 = Lea.if_(x, Lea.boolProb(1,2),
                    Lea.if_ (y, Lea.boolProb(1,7),
                                Lea.boolProb(1,5)) )
    assert z4.equiv(z1)
    assert (z4.given(y)).equiv(z1.given(y))

'''
TODO: remove
z0._factorsDict = {False: 3*420//5, True: 1*420//7}
z3._factorsDict = {False: 3*1, True: 4*420//2}
z0._cFactor = 1
z3._cFactor = 1

z0._factorsDict = {False: 420//5, True: 420//7}
z3._factorsDict = {False: 1*1, True: 420//2}
z0._cFactor = 1
z3._cFactor = 1

z0._factorsDict = {False: 420//5, True: 420//7}
z3._factorsDict = {False: 1, True: 4*420//2}
z0._cFactor = 3
z3._cFactor = 1

# this is OK for P(z3.given(y)) but NOK for P(z3)
# trick: P(z3.given(y==y)) gives the right result for P(z3)
z0._factorsDict = {False: 420//5, True: 420//7}
z3._factorsDict = {False: 1, True: 420//2}
z0._cFactor = 1
z3._cFactor = 1

# this is OK for P(z3.given(y)) but NOK for P(z3)
# trick: P(z3.given(y==y)) gives the right result for P(z3)
z0._factorsDict = {False: 420//5//3, True: 420//7//3}
z3._factorsDict = {False: 1, True: 420//2}
z0._cFactor = 3
z3._cFactor = 1


# this is OK for P(z3) but NOK for P(z3.given(y))
z0._factorsDict = {False: 420//5, True: 420//7}
z3._factorsDict = {False: 1, True: 4*420//2}
z0._cFactor = 1
z3._cFactor = 1
'''

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
