from lea import *
from lea.ilea import Ilea
from lea.toolbox import isclose
import pytest
from time import time
import random
import os
import tempfile

# setting this flag to True dramatically speeds up the test
EXACT_ALGO_ONLY = False

@pytest.fixture(scope="module")
def setup():
    set_prob_type('f')
    random.seed(0)   

# note; for testing the Monte-carlo algorithms, the tolerance for checking results may be lowered
# provided that the number of samples is increased (which slow down the tests)
def verify(query,exp_result,nb_samples=5000,nb_subsamples=10,nb_tries=None,exact_vars=None,exact_algo_only=False):
    # test exact algorithm (Statues)
    act_result = query.calc(algo=EXACT)
    assert act_result.equiv_f(exp_result,abs_tol=1e-12)
    if EXACT_ALGO_ONLY or exact_algo_only:
        return
    # test Monte-Carlo rejection sampling algorithm 
    act_result = query.calc(algo=MCRS,nb_samples=nb_samples)
    assert act_result.equiv_f(exp_result,abs_tol=2e-2)
    if isinstance(query,Ilea) or has_evidence():
        # test Monte-Carlo rejection sampling algorithm with subsampling 
        act_result = query.calc(algo=MCRS,nb_samples=nb_samples*nb_subsamples,nb_subsamples=nb_subsamples)
        assert act_result.equiv_f(exp_result,abs_tol=4e-2)
        # test Monte-Carlo likelihood weighting algorithm 
        act_result = query.calc(algo=MCLW,nb_subsamples=nb_samples)
        assert act_result.equiv_f(exp_result,abs_tol=2e-2)
        # test Monte-Carlo likelihood weighting algorithm with some variables treated with exact algorithm
        if exact_vars is not None:
            act_result = query.calc(algo=MCLW,nb_subsamples=nb_samples,exact_vars=exact_vars)
            assert act_result.equiv_f(exp_result,abs_tol=2e-2)
    elif exact_vars is not None:
        # test Monte-Carlo rejection sampling algorithm with some variables treated with exact algorithm
        act_result = query.calc(algo=MCEV,nb_subsamples=nb_samples,exact_vars=exact_vars)
        assert act_result.equiv_f(exp_result,abs_tol=1e-2)

# two fair dice
die1 = interval(1,6)
die2 = interval(1,6)
dice = die1 + die2

def test_fair_dice_1(setup):
    verify(die1,interval(1,6))

def test_fair_dice_2(setup):
    verify(dice,pmf({ 2: 1/36.,  3: 2/36.,  4: 3/36.,  5: 4/36.,  6: 5/36., 7: 6/36.,
                     12: 1/36., 11: 2/36., 10: 3/36.,  9: 4/36.,  8: 5/36., }))

def test_fair_dice_3(setup):
    verify(dice-die1-die2,0,exact_vars=(die1,))

def test_fair_dice_4(setup):
    verify(dice==die1,False,exact_vars=(die1,))

def test_fair_dice_5(setup):
    assert not has_evidence()
    verify(dice.given(die1==1),interval(2,7),exact_vars=(die1,)) 
    assert not has_evidence()

def test_fair_dice_5b(setup):
    assert not has_evidence()
    with evidence(die1==1):
        assert has_evidence()
        verify(dice,interval(2,7),exact_vars=(die1,))
        assert has_evidence()
    assert not has_evidence()

def test_fair_dice_6(setup):
    verify(die1.given(dice==3),interval(1,2),exact_vars=(die1,))

def test_fair_dice_6b(setup):
    assert not has_evidence()
    with evidence(dice==3):
        assert has_evidence()
        verify(die1,interval(1,2),exact_vars=(die1,))
        assert has_evidence()
    assert not has_evidence()

# test Bernoulli
b1 = bernoulli(0.5)
b2 = bernoulli(0.2)
b3 = bernoulli(0.1)
s = (b1-b2) * (1+b3)

def test_bernoulli_1(setup):
    verify(s, pmf({ -2: 0.01, -1: 0.09, 0: 0.5, 1: 0.36, 2: 0.04}), exact_vars=(b1,b2))

def test_bernoulli_2(setup):
    verify(s.given(b1==b2), 0, exact_vars=(b1,b2))

def test_bernoulli_3(setup):
    verify(s.given(b1!=b2), pmf({ -2: 0.02, -1: 0.18, 1: 0.72, 2: 0.08}), exact_vars=(b1,b2))

def test_bernoulli_4(setup):
    verify(b1.given(s==0), bernoulli(0.2), exact_vars=(b1,b2))

def test_bernoulli_5(setup):
    verify(s == b1*(1+b3) - b2 - b2*b3, True, exact_vars=(b1,b2))


# rain-sprinkler-grass Bayesian network
rain = event(0.20)
sprinkler = if_(rain, event(0.01),
                      event(0.40))
grass_wet = joint(sprinkler,rain).switch({ (False,False): False,
                                           (False,True ): event(0.80),
                                           (True ,False): event(0.90),
                                           (True ,True ): event(0.99)})

def test_bn1_1(setup):
    verify(rain, event(0.20), exact_vars=(rain,))

def test_bn1_2(setup):
    verify(sprinkler, event(0.322), exact_vars=(rain,))

def test_bn1_3(setup):
    verify(grass_wet, event(0.44838), exact_vars=(rain,))

def test_bn1_4(setup):
    verify(grass_wet & sprinkler&rain, event(0.00198), exact_vars=(rain,))

def test_bn1_5(setup):
    verify(~(grass_wet & sprinkler&rain), event(1. - 0.00198), exact_vars=(rain,))

def test_bn1_6(setup):
    verify(~(~grass_wet | ~sprinkler | ~rain), event(0.00198), exact_vars=(rain,))

def test_bn1_7(setup):
    verify(grass_wet.given(rain), event(0.8019), exact_vars=(rain,))

def test_bn1_7b(setup):
    assert not has_evidence()
    with evidence(rain):
        assert has_evidence()
        verify(grass_wet, event(0.8019), exact_vars=(rain,))
        assert has_evidence()
    assert not has_evidence()

def test_bn1_8(setup):
    verify(rain.given(grass_wet), event(0.3576876756322762), exact_vars=(rain,))

def test_bn1_9(setup):
    verify(joint(rain,grass_wet), pmf({(False, False) : 0.512,
                                       (False, True ) : 0.288,
                                       (True , False) : 0.03962,
                                       (True , True ) : 0.16038}), exact_vars=(rain,))

def test_bn1_10(setup):
    verify(joint(rain,grass_wet).given(rain==grass_wet), pmf({(False, False) : 0.7614741663940034,
                                                              (True , True ) : 0.23852583360599666}), exact_vars=(rain,))


# alarm Bayesian network
burglary   = event(0.001)
earthquake = event(0.002)
alarm = joint(burglary,earthquake).switch({ (True ,True ) : event(0.950),
                                            (True ,False) : event(0.940),
                                            (False,True ) : event(0.290),
                                            (False,False) : event(0.001) })
john_calls = alarm.switch({ True  : event(0.90),
                            False : event(0.05) })
mary_calls = alarm.switch({ True  : event(0.70),
                            False : event(0.01) })

def test_bn2_1(setup):
    verify(burglary.given(mary_calls | john_calls), event(0.014814211524985793), exact_algo_only=True)

def test_bn2_1b(setup):
    with evidence(mary_calls | john_calls):
        verify(burglary, event(0.014814211524985793), exact_algo_only=True)

def test_bn2_2(setup):
    verify(alarm, event(0.002516442), exact_algo_only=True)

def test_bn2_3(setup):
    verify(burglary & mary_calls, event(0.0006586138), exact_algo_only=True)

def test_bn2_4(setup):
    verify(~burglary & ~earthquake & alarm & john_calls & mary_calls, event(0.00062811126), exact_algo_only=True)

def test_bn2_5(setup):
    verify(burglary.given(mary_calls&john_calls), event(0.28417183536439294), exact_algo_only=True)

def test_bn2_6(setup):
    verify(burglary.given(mary_calls,john_calls), event(0.28417183536439294), exact_algo_only=True)

def test_bn2_6b(setup):
    with evidence(mary_calls,john_calls):
        verify(burglary, event(0.28417183536439294), exact_algo_only=True)

def test_bn2_6c(setup):
    with evidence(mary_calls):
        verify(burglary, event(0.05611745403891493), exact_algo_only=True)
        with evidence(john_calls):
            verify(burglary, event(0.28417183536439294), exact_algo_only=True)
        verify(burglary, event(0.05611745403891493), exact_algo_only=True)
    verify(burglary, event(0.001), exact_algo_only=True)

def test_bn2_7(setup):
    verify(mary_calls.given(burglary), event(0.6586138), exact_algo_only=True)
    
def test_bn2_7b(setup):
    with evidence(bindings={burglary:True}):
        verify(mary_calls, event(0.6586138), exact_vars=(alarm,))
    verify(mary_calls, event(0.011736344979999999), exact_algo_only=True)

# BIF fil found on https://www.bnlearn.com/bnrepository/discrete-small.html#earthquake
# changed Burglary probability to 0.001 instead of 0.01
earthqiake_bif_content = """
network unknown {
}
variable Burglary {
  type discrete [ 2 ] { True, False };
}
variable Earthquake {
  type discrete [ 2 ] { True, False };
}
variable Alarm {
  type discrete [ 2 ] { True, False };
}
variable JohnCalls {
  type discrete [ 2 ] { True, False };
}
variable MaryCalls {
  type discrete [ 2 ] { True, False };
}
probability ( Burglary ) {
  table 0.001, 0.999;
}
probability ( Earthquake ) {
  table 0.002, 0.998;
}
probability ( Alarm | Burglary, Earthquake ) {
  (True, True) 0.95, 0.05;
  (False, True) 0.29, 0.71;
  (True, False) 0.94, 0.06;
  (False, False) 0.001, 0.999;
}
probability ( JohnCalls | Alarm ) {
  (True) 0.9, 0.1;
  (False) 0.05, 0.95;
}
probability ( MaryCalls | Alarm ) {
  (True) 0.7, 0.3;
  (False) 0.01, 0.99;
}
"""

def test_read_bif_file(setup):
    bif_filename = os.path.join(tempfile.mkdtemp(),"earthquake.bif")
    with open(bif_filename,'w') as f:
        f.write(earthqiake_bif_content)
    var_names = read_bif_file(bif_filename,globals())
    assert frozenset(var_names) == frozenset(('Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls'))
    assert Burglary.equiv(event(0.001))
    assert isclose(P(MaryCalls.given(Alarm)), 0.7)
    assert isclose(P(MaryCalls.given(Burglary)), 0.6586138)
    assert isclose(P(Burglary.given(Alarm)), 0.373551228281836)
    assert isclose(P(Alarm.given(MaryCalls)), 0.1500901177497596)
    assert isclose(P(Burglary.given(MaryCalls | JohnCalls)), 0.014814211524985793)
    assert isclose(P(Burglary.given(MaryCalls & JohnCalls)), 0.28417183536439294)
    assert isclose(P(Alarm), 0.002516442)
    assert isclose(P(Burglary & MaryCalls), 0.0006586138000000001)
    assert isclose(P(~Burglary & ~Earthquake & Alarm & JohnCalls & MaryCalls), 0.00062811126)
