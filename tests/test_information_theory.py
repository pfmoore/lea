import lea
import pytest
from lea.toolbox import isclose, log2

# All tests are made using float representation
@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('f')
    

def test_information_1(setup):
    flip = lea.vals('head','tail')
    assert isclose(flip.information_of('head'), 1.0)
    assert isclose(flip.information_of('tail'), 1.0)
    flip_u = lea.pmf({'head': 1./4., 'tail': 3./4.})    
    assert isclose(flip_u.information_of('head'), 2.0)
    assert isclose(flip_u.information_of('tail'), 0.4150374992788437)
    rain = lea.event(1./8.)
    assert isclose(rain.information, 3.0)  
    assert isclose((flip_u=='head').information, 2.0)

def test_information_2(setup):
    flip = lea.event(0.5)
    assert isclose(flip.information, 1.0)
    assert isclose(flip.information_of(True), 1.0)
    assert isclose(flip.information_of(False), 1.0)
    die = lea.interval(1,6)
    assert isclose(die.information_of(1), -log2(1./6.))
    assert isclose(die.information_of(6), -log2(1./6.))
    d = lea.pmf({"A": 0.5, "B": 0.25, "C": 0.25})
    assert isclose(d.information_of("A"), 1.0)   
    assert isclose(d.information_of("B"), 2.0)
    with pytest.raises(lea.Lea.Error):
        d.information_of("D")

def test_entropy_1(setup):
    flip = lea.vals('head','tail')
    assert isclose(flip.entropy, 1.0)
    flip_u = lea.pmf({'head': 1./4., 'tail': 3./4.})
    assert isclose(flip_u.entropy, 0.8112781244591328)
    ball = lea.pmf({ 'Bx': 62, 'Rx': 1, 'Ry': 1 })
    color = ball[0]
    mark = ball[1]
    assert isclose(ball.entropy, 0.23187232431271465)
    assert isclose(ball.given(mark=='x').entropy, 0.11759466565886476)
    assert isclose(ball.given(mark=='y').entropy, 0.0)
    assert isclose(ball.given(color=='B').entropy, 0.0)
    assert isclose(ball.given(color=='R').entropy, 1.0)

def test_entropy_2(setup):
    flip = lea.event(0.5)
    assert isclose(flip.entropy, 1.0)
    assert isclose(flip.entropy, 1.0)
    die = lea.interval(1,6)
    assert isclose(die.entropy, -log2(1./6.))
    assert  isclose((die+2).entropy, -log2(1./6.))
    d = lea.pmf({"A": 0.5, "B": 0.25, "C": 0.25})
    assert isclose(d.entropy, 1.5)

def test_rel_entropy(setup):
    flip = lea.event(0.5)
    assert isclose(flip.rel_entropy, 1.0)
    assert isclose((~flip).rel_entropy, 1.0)
    die = lea.interval(1,6)
    assert isclose(die.rel_entropy, 1.0)
    assert isclose((die+2).rel_entropy, 1.0)
    d = lea.pmf({"A": 0.5, "B": 0.25, "C": 0.25})
    assert isclose(d.rel_entropy, 1.5/log2(3))

def test_mutual_information_1(setup):
    ball = lea.pmf({ 'Bx': 62, 'Rx': 1, 'Ry': 1 })
    color = ball[0]
    mark = ball[1]
    assert isclose(lea.mutual_information(color,mark), 0.08486507530476972)
    assert isclose(lea.mutual_information(color,ball), 0.20062232431271465)
    assert isclose(lea.mutual_information(mark,ball), 0.11611507530476972)
    flip = lea.event(0.5)
    assert lea.mutual_information(ball,flip) == 0.0
    
def test_joint_entropy(setup):
    ball = lea.pmf({ 'Bx': 62, 'Rx': 1, 'Ry': 1 })
    color = ball[0]
    mark = ball[1]
    assert isclose(lea.joint_entropy(mark,color), 0.23187232431271465)
    flip = lea.event(0.5)
    assert isclose(lea.joint_entropy(mark,flip), 1.1161150753047697)

def test_cond_entropy(setup):
    ball = lea.pmf({ 'Bx': 62, 'Rx': 1, 'Ry': 1 })
    color = ball[0]
    mark = ball[1]
    assert isclose(ball.cond_entropy(mark), 0.11575724900794494)
    assert isclose(mark.cond_entropy(ball), 0.0)
    flip = lea.event(0.5)
    assert isclose(ball.cond_entropy(flip), 0.23187232431271465)
    assert isclose(flip.cond_entropy(ball), 1.0)
    assert isclose(flip.cond_entropy(mark), 1.0)
    assert isclose(flip.cond_entropy(flip), 0.0)
    assert isclose(ball.cond_entropy(ball), 0.0)

def test_cross_entropy(setup):
    ball = lea.pmf({ 'Bx': 62, 'Rx': 1, 'Ry': 1 })
    color = ball[0]
    estimated_color1 = lea.pmf({ 'B': 20, 'R': 12 })
    assert isclose(color.cross_entropy(estimated_color1), 0.7011020799303316)
    estimated_color2 = lea.pmf({ 'B': 30, 'R': 2 })
    assert isclose(color.cross_entropy(estimated_color2), 0.2151997355042477)
    assert isclose(color.cross_entropy(color), 0.20062232431271465)
    
def test_kl_divergence(setup):
    ball = lea.pmf({ 'Bx': 62, 'Rx': 1, 'Ry': 1 })
    color = ball[0]
    estimated_color1 = lea.pmf({ 'B': 20, 'R': 12 })
    assert isclose(color.kl_divergence(estimated_color1), 0.5004797556176169)
    estimated_color2 = lea.pmf({ 'B': 30, 'R': 2 })
    assert isclose(color.kl_divergence(estimated_color2), 0.014577411191533052)
    assert isclose(color.kl_divergence(color), 0.0)
    b1 = lea.binom(2,0.3)
    b2 = lea.binom(3,0.2)
    bs = b1 + b2
    assert isclose(b1.kl_divergence(b1), 0.0)
    assert isclose(bs.kl_divergence(bs), 0.0)

def test_relations(setup):
    b1 = lea.binom(2,0.3)
    b2 = lea.binom(3,0.2)
    bs = b1 + b2
    mi = lea.mutual_information(b1,bs)
    assert isclose(mi, lea.mutual_information(bs,b1))
    assert isclose(mi, bs.entropy - bs.cond_entropy(b1))
    assert isclose(mi, b1.entropy - b1.cond_entropy(bs))
    assert isclose(mi, b1.entropy + bs.entropy - lea.joint_entropy(b1,bs))
    assert isclose(mi, lea.joint(b1,bs).entropy - b1.cond_entropy(bs) - bs.cond_entropy(b1) )
    assert isclose(lea.joint_entropy(b1,bs), lea.joint(b1,bs).entropy)
    ball = lea.pmf({ 'Bx': 62, 'Rx': 1, 'Ry': 1 })
    color = ball[0]
    mark = ball[1]
    assert ball.cond_entropy(b1) == ball.entropy
    assert ball.cond_entropy(bs) == ball.entropy
    assert b1.cond_entropy(ball) == b1.entropy
    assert bs.cond_entropy(mark) == bs.entropy
    assert isclose(lea.joint_entropy(mark,color), lea.joint(mark,color).entropy)
    flip = lea.event(0.5)
    assert isclose(lea.joint_entropy(ball,flip), ball.entropy + flip.entropy)
    assert isclose(lea.joint_entropy(mark,flip), mark.entropy + flip.entropy)
    assert isclose(lea.joint_entropy(color,flip), color.entropy + flip.entropy)
    assert isclose(lea.joint_entropy(ball,bs), ball.entropy + bs.entropy)
    assert isclose(lea.joint_entropy(bs,ball), lea.joint_entropy(ball,bs))
    assert isclose(color.cross_entropy(color), color.entropy)
    assert isclose(b1.cross_entropy(b1), b1.entropy)
    assert isclose(bs.cross_entropy(bs), bs.entropy)
    estimated_color1 = lea.pmf({ 'B': 20, 'R': 12 })
    assert isclose(color.kl_divergence(estimated_color1), color.cross_entropy(estimated_color1)-color.entropy)
