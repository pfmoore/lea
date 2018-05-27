import lea
from lea import P

# All tests are made using fraction representation, in order to ease comparison
lea.set_prob_type('r')

def test_compare_self():
    dist = lea.pmf(((1,1), (2,2), (3,3), (4,4)))
    assert P(dist == dist) == 1
    assert P(dist != dist) == 0
    assert P(dist > dist) == 0
    assert P(dist < dist) == 0
    assert P(dist >= dist) == 1
    assert P(dist <= dist) == 1

def test_comparison_equivs():
    dist1 = lea.pmf(((1,1), (2,2), (3,3), (4,4)))
    dist2 = lea.pmf(((1,4), (2,3), (3,2), (4,1)))
    assert P((dist1 > dist2) == ~(dist1 <= dist2)) == 1
    assert P((dist1 >= dist2) == ~(dist1 < dist2)) == 1
    assert P((dist1 == dist2) == ~(dist1 != dist2)) == 1

def test_multiply_sum():
    dist1 = lea.pmf(((1,1), (2,2), (3,3), (4,4)))
    dist2 = lea.pmf(((1,4), (2,3), (3,2), (4,1)))
    dist3 = lea.pmf(((1,1), (2,1), (3,1), (4,1)))
    assert P((dist1 * (dist2 + dist3)) == ((dist1 * dist2) + (dist1 * dist3))) == 1
