import lea
from lea.prob_fraction import ProbFraction as PF
from lea.leaf import D6
import pytest

# All tests are made using fraction representation, in order to ease comparison
#lea.set_prob_type('r')

@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('r')

def test_times_commutativity(setup):
    """See issue #3 on bitbucket"""
    lr = D6.times(2) + D6
    rl = D6 + D6.times(2)
    assert lr.equiv(rl)

def test_withprob(setup):
    """See issues #8 and #15 on Bitbucket"""
    dU = D6.given_prob(D6>=5,PF(1,2))
    assert dU.p(5) == PF(2,8)
    assert dU.equiv(lea.pmf(((1,1),(2,1),(3,1),(4,1),(5,2),(6,2))))

def test_infeasible_cpt(setup):
    """See issue #13 on Bitbucket"""
    a = lea.vals(*"xy")
    always_s = lea.cpt((a == a, 's'), (None, 't'))
    never_s = lea.cpt((a != a, 's'), (None, 't'))
    assert always_s.equiv('s')
    assert never_s.equiv('t')

def test_cpt_refactoring(setup):
    """See issues #9 and #11 on Bitbucket"""
    x = lea.event(PF(1,3))
    y = lea.event(PF(1,4))
    z1 = lea.cpt(
            (~x & ~y, lea.event(PF(1,5))),
            (~x &  y, lea.event(PF(1,7))),
            ( x & ~y, lea.event(PF(1,2))),
            ( x &  y, lea.event(PF(1,2))),
        )
    z2 = lea.cpt(
            (~x & ~y, lea.event(PF(1,5))),
            (~x &  y, lea.event(PF(1,7))),
            ( x     , lea.event(PF(1,2))),
        )
    z0 = lea.cpt(
            (~y, lea.event(PF(1,5))),
            ( y, lea.event(PF(1,7))),
        )
    z3 = lea.cpt(
            (~x, z0),
            ( x, lea.event(PF(1,2))),
        )
    assert z2.equiv(z1)
    assert z3.equiv(z1)

def test_draw_nonuniform(setup):
    """See issue #19 on bitbucket"""
    h = lea.pmf((("A",3),("B",2),("C",1)))
    expected = lea.pmf((
            (('A', 'B'), 20),
            (('A', 'C'), 10),
            (('B', 'A'), 15),
            (('B', 'C'),  5),
            (('C', 'A'),  6),
            (('C', 'B'),  4),
        ))
    assert h.draw(2).equiv(expected)

def test_joint(setup):
    """See issue #20 on bitbucket"""
    joint = lea.pmf((((1,2),10), ((1,3),9), ((2,2),8))).as_joint('A','B')
    assert joint.p((1,2)) == PF(10,27)
    assert float(joint.p((1,2))) == 10.0 / 27.0

def test_check_bool():
    """See issue #24 on bitbucket"""
    mixed = lea.vals(True,False,12)
    with pytest.raises(lea.Lea.Error):
        mixed.is_true()
    with pytest.raises(lea.Lea.Error):
        mixed.is_feasible()
    with pytest.raises(lea.Lea.Error):
        mixed.P
    with pytest.raises(lea.Lea.Error):
        mixed.Pf

def test_given_times(setup):
    """See issue #28 on bitbucket"""
    flip = lea.vals(0,1)
    flip4 = flip.times(4)
    expected = lea.pmf(((0,1), (1,4), (2,6)))
    assert flip4.given(flip4<=2).equiv(expected)


def test_sympy_expressions(setup):
    """No issue number; make sure sympy classes work."""
    try:
        from sympy.core.numbers import Rational
        v = Rational(1, 4)
        probs = {
          'A': v,
          'B': None,
        }
        _ = lea.pmf(probs, prob_type = -1)
    except ImportError:
        pass

def test_symbol_prob_type(setup):
    """No issue number; make sure 's' prob_type works."""
    try:
        import sympy  # failure skips the test entirely
        probs = {
          'A': 1,
          'B': None,
        }
        _ = lea.pmf(probs, prob_type = 's')
    except ImportError:
        pass


def test_symbol_prob_type_with_identifier_check(setup):
    """
    No issue number; make sure 's' prob_type works when values might be
    identifiers.
    """
    try:
        import sympy  # failure skips the test entirely
        probs = {
          'A': 'p',
          'B': 'q',
        }
        _ = lea.pmf(probs, prob_type = 's')
    except ImportError:
        pass

def test_symbol_prob_type_with_identifier_check_2(setup):
    """
    No issue number; make sure automatic type detection works when values
    might be identifiers (without specifying prob_type='s')
    """
    try:
        import sympy  # failure skips the test entirely
        # reset permanent prob_type to automatic type detection mode
        lea.set_prob_type('x')
        probs = {
          'A': 'p',
          'B': 'q',
        }
        _ = lea.pmf(probs)
    except ImportError:
        pass

