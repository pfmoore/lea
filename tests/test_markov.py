import lea
import pytest

from lea import markov 
from lea.exceptions import LeaError
from lea.toolbox import isclose

@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('f')

def create_market_mc():
    """create and return a MC"""
    # test made with float representation for probabilities
    lea.set_prob_type('f')
    return markov.chain_from_matrix(
                  ( 'BULL', 'BEAR', 'STAG'  ),
        ( 'BULL', ( 0.900 , 0.075 , 0.025  )),
        ( 'BEAR', ( 0.150 , 0.800 , 0.050  )),
        ( 'STAG', ( 0.250 , 0.250 , 0.500  )))

def test_markov_create(setup):
    """Check that a MC can be created (markov.chain_from_matrix function) and displayed"""
    market = create_market_mc()
    assert str(market) == 'BULL\n  -> BULL : 0.9\n  -> BEAR : 0.075\n  -> STAG : 0.025\nBEAR\n  -> BULL : 0.15\n  -> BEAR : 0.8\n  -> STAG : 0.05\nSTAG\n  -> BULL : 0.25\n  -> BEAR : 0.25\n  -> STAG : 0.5'

def test_markov_states(setup):
    """Check that markov.Chain.state attribute is correct"""
    market = create_market_mc()
    assert market.states == ('BULL', 'BEAR', 'STAG')

def test_markov_state(setup):
    """Check that markov.Chain.state attribute is correct"""
    market = create_market_mc()
    assert market.state.equiv_f(lea.pmf({'BULL': 1./3., 'BEAR': 1./3., 'STAG': 1./3.,}))

def test_markov_get_states(setup):
    """Check that markov.Chain.get_states method is correct"""
    market = create_market_mc()
    (bull_state,bear_state,stag_state) = market.get_states()
    assert bull_state.equiv_f(lea.pmf({'BULL': 1.}))
    assert bear_state.equiv_f(lea.pmf({'BEAR': 1.}))
    assert stag_state.equiv_f(lea.pmf({'STAG': 1.}))

def test_markov_next_state(setup):
    """Check that markov.Chain.next_state method is correct"""
    market = create_market_mc()
    (bull_state,bear_state,stag_state) = market.get_states()
    assert bear_state.next_state().equiv_f(lea.pmf({'BULL': 0.150, 'BEAR': 0.800, 'STAG': 0.050}))
    assert bear_state.next_state(1).equiv_f(lea.pmf({'BULL': 0.150, 'BEAR': 0.800, 'STAG': 0.050}))
    assert bear_state.next_state(1,keeps_dependency=False).equiv_f(lea.pmf({'BULL': 0.150, 'BEAR': 0.800, 'STAG': 0.050}))
    assert bear_state.next_state(3).equiv_f(lea.pmf({'BULL': 0.35750, 'BEAR': 0.56825, 'STAG': 0.07425}))
    assert bear_state.next_state(3,keeps_dependency=False).equiv_f(lea.pmf({'BULL': 0.35750, 'BEAR': 0.56825, 'STAG': 0.07425}))
    # stationary distribution
    assert bear_state.next_state(100).equiv_f(lea.pmf({'BULL': 0.6250, 'BEAR': 0.3125, 'STAG': 0.0625}))
    fp_state = market.get_state(lea.pmf({ 'BULL': 0.6250,
                                          'BEAR': 0.3125,
                                          'STAG': 0.0625 }))
    assert fp_state.next_state().equiv_f(fp_state)
    assert isclose(lea.mutual_information(market.state,market.state.next_state()),0.5321367231080285)
    assert isclose(lea.mutual_information(market.state,market.state.next_state(keeps_dependency=False)),0.0)
    with pytest.raises(LeaError):
        bear_state.next_state(0)
    with pytest.raises(LeaError):
        bear_state.next_state(-1)

def test_markov_state_given(setup):
    """Check that markov.Chain.state_given method is correct"""
    market = create_market_mc()
    (bull_state,bear_state,stag_state) = market.get_states()
    assert market.state_given(market.state=='BULL').equiv_f(bull_state)
    assert market.state_given(market.state[0]=='B').equiv_f(lea.pmf({'BULL': 0.5, 'BEAR': 0.5}))
    assert market.state_given((market.state[0]=='B')&(market.state!='BEAR')).equiv_f(bull_state)
    with pytest.raises(LeaError):
        market.state_given(False).calc()
    with pytest.raises(LeaError):
        market.state_given(market.state=='XXX').calc()

def test_markov_next_state_given(setup):
    """Check that markov.Chain.next_state_given method is correct"""
    market = create_market_mc()
    (bull_state,bear_state,stag_state) = market.get_states()
    assert market.next_state_given(market.state=='BEAR').equiv_f(lea.pmf({'BULL': 0.150, 'BEAR': 0.800, 'STAG': 0.050}))
    assert market.next_state_given(market.state=='BEAR',1).equiv_f(lea.pmf({'BULL': 0.150, 'BEAR': 0.800, 'STAG': 0.050}))
    assert market.next_state_given(market.state=='BEAR',3).equiv_f(lea.pmf({'BULL': 0.35750, 'BEAR': 0.56825, 'STAG': 0.07425}))
    with pytest.raises(LeaError):
        market.next_state_given(market.state=='BEAR',0)
    with pytest.raises(LeaError):
        market.next_state_given(market.state=='BEAR',-1)
    with pytest.raises(LeaError):
        market.next_state_given(market.state=='XXX').calc()

def test_markov_matrix(setup):
    """Check that markov.Chain.matrix method is correct"""
    market = create_market_mc()
    assert market.states == ('BULL', 'BEAR', 'STAG')
    assert market.matrix() == ((0.9, 0.075, 0.025), (0.15, 0.8, 0.05), (0.25, 0.25, 0.5))
    assert market.matrix(from_states=('BULL','STAG')) == ((0.9, 0.075, 0.025), (0.25, 0.25, 0.5))
    assert market.matrix(from_states=('BULL','STAG'),to_states=('BULL','STAG')) == ((0.9, 0.025), (0.25, 0.5))
    assert market.matrix(from_states=('BEAR',),to_states=('BEAR',)) == ((0.8,),)
    assert market.matrix(from_states=()) == ()
    assert market.matrix(to_states=()) == ((), (), ())
    assert market.matrix(from_states=(),to_states=()) == ()

def test_markov_chain_from_seq(setup):
    """Check that markov.chain_from_seq function is correct"""
    mc = markov.chain_from_seq(('A','A','B','B','B','A','C','B','C','A','A','A','B','A','A','B'))
    assert mc.states == ('A', 'B', 'C')
    assert mc.matrix() == ((0.5, 0.375, 0.125), (0.4, 0.4, 0.2), (0.5, 0.5, 0.0))
    # last given state 'D' has no transition: then,'D' -> 'D' is assumed
    mc = markov.chain_from_seq(('A','A','B','B','B','A','C','B','C','A','A','A','B','A','A','D'))
    assert mc.states == ('A', 'B', 'C', 'D')
    assert mc.matrix() == ((0.5, 0.25, 0.125, 0.125), (0.4, 0.4, 0.2, 0.0), (0.5, 0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

def test_markov_absorbing_mc_info(setup):
    """Check that markov.chain_from_seq function is correct"""
    market = create_market_mc()
    (is_absorbing1, transient_states1, absorbing_states1, q_matrix1, r_matrix1, n_matrix1) = market.absorbing_mc_info()
    assert not is_absorbing1
    assert transient_states1 == ()
    assert absorbing_states1 == ()
    assert q_matrix1 == ()
    assert r_matrix1 == ()
    assert n_matrix1 is None
    # BEAR is absorbing
    market2 = markov.chain_from_matrix(
                  ( 'BULL', 'BEAR', 'STAG'  ),
        ( 'BULL', ( 0.900 , 0.075 , 0.025  )),
        ( 'BEAR', ( 0.000 , 0.100 , 0.000  )),
        ( 'STAG', ( 0.250 , 0.250 , 0.500  )))
    (is_absorbing2, transient_states2, absorbing_states2, q_matrix2, r_matrix2, n_matrix2) = market2.absorbing_mc_info()
    assert is_absorbing2
    assert transient_states2 == ('BULL', 'STAG')
    assert absorbing_states2 == ('BEAR',)
    assert q_matrix2 == ((0.9, 0.025), (0.25, 0.5))
    assert r_matrix2 == ((0.075,), (0.25,))
    assert n_matrix2 is None
    # BEAR is absorbing but unreachable 
    market3 = markov.chain_from_matrix(
                  ( 'BULL', 'BEAR', 'STAG'  ),
        ( 'BULL', ( 0.900 , 0.000 , 0.100  )),
        ( 'BEAR', ( 0.000 , 0.100 , 0.000  )),
        ( 'STAG', ( 0.200 , 0.000 , 0.800  )))
    (is_absorbing3, transient_states3, absorbing_states3, q_matrix3, r_matrix3, n_matrix3) = market3.absorbing_mc_info()
    assert not is_absorbing3
    assert transient_states3 == ()
    assert absorbing_states3 == ('BEAR',)
    assert q_matrix3 == ()
    assert r_matrix3 == ()
    assert n_matrix3 is None

def test_markov_state_dependencies(setup):
    """ Check dependencies between Markov states (see #63) """
    weather = markov.chain_from_matrix(('sunny','rainy'),
                          ('sunny',(  0.95 ,  0.05 )),
                          ('rainy',(  0.80 ,  0.20 )))
    today = weather.next_state('sunny',1000)
    tomorrow = today.next_state()
    assert today.is_dependent_of(tomorrow)
    assert tomorrow.is_dependent_of(today)
    assert isclose(lea.mutual_information(today,tomorrow), 0.010740523089006304)
    tomorrow2 = today.next_state(keeps_dependency=False)
    assert not today.is_dependent_of(tomorrow2)
    assert not tomorrow2.is_dependent_of(today)
    assert lea.mutual_information(today,tomorrow2) == 0.0
