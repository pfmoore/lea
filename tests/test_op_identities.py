from lea import Lea, P

def test_compare_self():
    dist = Lea.fromValFreqs((1,1), (2,2), (3,3), (4,4))
    assert P(dist == dist) == 1
    assert P(dist != dist) == 0
    assert P(dist > dist) == 0
    assert P(dist < dist) == 0
    assert P(dist >= dist) == 1
    assert P(dist <= dist) == 1

def test_comparison_equivs():
    dist1 = Lea.fromValFreqs((1,1), (2,2), (3,3), (4,4))
    dist2 = Lea.fromValFreqs((1,4), (2,3), (3,2), (4,1))
    assert P((dist1 > dist2) == ~(dist1 <= dist2)) == 1
    assert P((dist1 >= dist2) == ~(dist1 < dist2)) == 1
    assert P((dist1 == dist2) == ~(dist1 != dist2)) == 1

def test_multiply_sum():
    dist1 = Lea.fromValFreqs((1,1), (2,2), (3,3), (4,4))
    dist2 = Lea.fromValFreqs((1,4), (2,3), (3,2), (4,1))
    dist3 = Lea.fromValFreqs((1,1), (2,1), (3,1), (4,1))
    assert P((dist1 * (dist2 + dist3)) == ((dist1 * dist2) + (dist1 * dist3))) == 1
