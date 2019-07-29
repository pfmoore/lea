import lea
import pytest
import math
from lea.toolbox import isclose
import random

# All tests are made using float representation, in order to ease comparison with results found in litterature
@pytest.fixture(scope="module")
def setup():
    lea.set_prob_type('f')

def test_candy(setup):
    # reference: Artifical Intelligence: a Modern Approach (2nd edition) - p.729
    # observed data   
    (T,F) = (True, False)
    candy_jpd = lea.pmf({ ('red',T,'cherry'): 273, ('red',F,'cherry'):  93, ('green',T,'cherry'): 104, ('green',F,'cherry'):  90,
                          ('red',T,'lime'  ):  79, ('red',F,'lime'  ): 100, ('green',T,'lime'  ):  94, ('green',F,'lime'  ): 167})
    # 'w': wrapper, 'h': holes, 'f': flavor
    obs_candy = candy_jpd.as_joint('w','h','f')
    # prior model - iteration 0
    bag0 = lea.pmf({1: 0.6, 2: 0.4})
    wb1_0 = lea.pmf({'red': 0.6, 'green': 0.4})
    wb2_0 = lea.pmf({'red': 0.4, 'green': 0.6})
    w0 = bag0.switch({1: wb1_0, 2: wb2_0})
    hb1_0 = lea.event(0.6)
    hb2_0 = lea.event(0.4)
    h0 = bag0.switch({1: hb1_0, 2: hb2_0})
    fb1_0 = lea.pmf({'cherry': 0.6, 'lime': 0.4})
    fb2_0 = lea.pmf({'cherry': 0.4, 'lime': 0.6})
    f0 = bag0.switch({1: fb1_0, 2: fb2_0})
    candy0 = lea.joint(w0,h0,f0)
    model_dict = candy0.learn_by_em(obs_candy,nb_steps=1)
    (candy1,bag1,w1,h1,f1,wb1_1,wb2_1,hb1_1,hb2_1,fb1_1,fb2_1) = \
        tuple(model_dict[k] for k in (candy0,bag0,w0,h0,f0,wb1_0,wb2_0,hb1_0,hb2_0,fb1_0,fb2_0))
    # check parameters of new model using new pmf variables
    assert isclose(lea.P(bag1==1)                      , 0.6124, abs_tol=1e-4)
    assert isclose(lea.P(fb1_1=='cherry')              , 0.6684, abs_tol=1e-4)
    assert isclose(lea.P(wb1_1=='red')                 , 0.6483, abs_tol=1e-4)
    assert isclose(lea.P(hb1_1)                        , 0.6558, abs_tol=1e-4)
    assert isclose(lea.P(fb2_1=='cherry')              , 0.3887, abs_tol=1e-4)
    assert isclose(lea.P(wb2_1=='red')                 , 0.3817, abs_tol=1e-4)
    assert isclose(lea.P(hb2_1)                        , 0.3827, abs_tol=1e-4)
    # check parameters of new model using new CPT variables
    assert isclose(lea.P((f1=='cherry').given(bag1==1)), 0.6684, abs_tol=1e-4)
    assert isclose(lea.P((w1=='red').given(bag1==1))   , 0.6483, abs_tol=1e-4)
    assert isclose(lea.P(h1.given(bag1==1))            , 0.6558, abs_tol=1e-4)
    assert isclose(lea.P((f1=='cherry').given(bag1==2)), 0.3887, abs_tol=1e-4)
    assert isclose(lea.P((w1=='red').given(bag1==2))   , 0.3817, abs_tol=1e-4)
    assert isclose(lea.P(h1.given(bag1==2))            , 0.3827, abs_tol=1e-4)
    # check log-likelihood of initial model
    assert isclose(-1000 * obs_candy.cross_entropy(candy0) * math.log(2), -2044, abs_tol=1)
    # check log-likelihood of model after one EM step
    assert isclose(-1000 * obs_candy.cross_entropy(candy1) * math.log(2), -2021, abs_tol=1)
    # perform 10 steps of EM algorithm using learn_by_em method
    model_dict10 = candy0.learn_by_em(obs_candy,nb_steps=10)
    (candy10,bag10,w10,h10,f10) = tuple(model_dict10[var] for var in (candy0,bag0,w0,h0,f0))
    # check log-likelihood of model after 10 EM steps
    assert isclose(-1000 * obs_candy.cross_entropy(candy10) * math.log(2), -1982, abs_tol=1)
    # perform 10 steps of EM algorithm using the gen_em_steps generator method
    candy_gen_steps = candy0.gen_em_steps(obs_candy)
    for _ in range(10):
       model_dictx = next(candy_gen_steps)
    candyx = model_dictx[candy0]
    assert isclose(-1000 * obs_candy.cross_entropy(candyx) * math.log(2), -1982, abs_tol=1)
    assert candyx.equiv_f(candy10)
    # perform x steps of EM algorithm using learn_by_em method, until kl divergence is <= 1e-3
    model_dicty = candy0.learn_by_em(obs_candy,max_kld=1e-3)
    candyy = model_dicty[candy0]
    assert isclose(obs_candy.kl_divergence(candyy),0.0,abs_tol=1e-3)

def test_coins(setup):
    # http://ibug.doc.ic.ac.uk/media/uploads/documents/expectation_maximization-1.pdf
    ha0 = 0.6
    hb0 = 0.5
    coins_A0 = lea.binom(10, ha0)
    coins_B0 = lea.binom(10, hb0)
    # note that the example does not cover the coin distribution, we assume fixed 50-50 (not to be learned) 
    coin_type0 = lea.vals("A","B")
    coins0 = coin_type0.switch( { "A" : coins_A0,
                                  "B" : coins_B0})
    obs_coins = lea.vals(5,9,8,4,7)
    # perform one single step of EM algorithm, keep internal binomial variables hidden
    model_dict = coins0.learn_by_em(obs_coins,fixed_vars=(coin_type0,),nb_steps=1)
    coins1 = model_dict[coins0]
    coin_type1 = model_dict[coin_type0]
    # check that coin_type has not changed
    assert coin_type1 is coin_type0
    # check indirectly parameters of new model, without accessing binomial parameters,
    # knowing that p = mean/n in a binomial  
    assert isclose(coins1.given(coin_type1=='A').mean/10, 0.71, abs_tol=1e-2)
    assert isclose(coins1.given(coin_type1=='B').mean/10, 0.58, abs_tol=1e-2)
    # perform one single step of EM algorithm, getting the new binomial variables 
    model_dict = coins0.learn_by_em(obs_coins,fixed_vars=(coin_type0,),nb_steps=1)
    coins_A1 = model_dict[coins_A0]
    coins_B1 = model_dict[coins_B0]
    assert isclose(coins_A1.p, 0.71, abs_tol=1e-2)
    assert isclose(coins_B1.p, 0.58, abs_tol=1e-2)

def test_mixt_binom(setup):
    # true model (supposing unknown)
    a = lea.binom(10,0.4)
    b = lea.binom(16,0.7)
    c = lea.pmf({'A': 0.2, 'B': 0.8})
    x = c.switch({'A': a, 'B': b})
    # observed data, based on 100,000 samples
    random.seed(0)
    obs_x = lea.vals(*x.random(100000))
    # initial model, with estimated parameters
    a0 = lea.binom(10,0.5)
    b0 = lea.binom(16,0.5)
    c0 = lea.pmf({'A': 0.4, 'B': 0.6})
    x0 = c0.switch({'A': a0, 'B': b0})
    # perform 50 steps of EM algorithm 
    model_dict = x0.learn_by_em(obs_x,nb_steps=50)
    (x1,a1,b1,c1) = (model_dict[var] for var in (x0,a0,b0,c0))
    assert isclose(a1.p,a.p,abs_tol=1e-2)
    assert isclose(b1.p,b.p,abs_tol=1e-2)
    assert isclose(c1.p("A"),c.p("A"),abs_tol=1e-2)
    assert x1.equiv_f(x,abs_tol=1e-2)
    assert isclose(obs_x.kl_divergence(x1),0.0,abs_tol=1e-3)

def test_mixt_binom_poisson(setup):
    # true model (supposing unknown)
    a = lea.binom(10,0.4)
    b = lea.poisson(8)
    c = lea.pmf({'A': 0.2, 'B': 0.8})
    x = c.switch({'A': a, 'B': b})
    # observed data, based on 100,000 samples
    random.seed(0)
    obs_x = lea.vals(*x.random(100000))
    # initial model, with estimated parameters
    a0 = lea.binom(10,0.5)
    b0 = lea.poisson(10)
    c0 = lea.pmf({'A': 0.4, 'B': 0.6})
    x0 = c0.switch({'A': a0, 'B': b0})
    # perform 50 steps of EM algorithm 
    model_dict = x0.learn_by_em(obs_x,nb_steps=50)
    (x1,a1,b1,c1) = (model_dict[var] for var in (x0,a0,b0,c0))
    assert isclose(a1.p,a.p,abs_tol=1e-2)
    assert isclose(b1._mean,b._mean,abs_tol=1e-1)
    assert isclose(c1.p("A"),c.p("A"),abs_tol=1e-2)
    assert x1.equiv_f(x,abs_tol=1e-2)
    assert isclose(obs_x.kl_divergence(x1),0.0,abs_tol=1e-3)

def test_add(setup):
    a = lea.pmf({0: 0.5, 2: 0.3, 4: 0.2})
    b = lea.pmf({0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2})
    x = a + b
    # observed data, based on 1,000,000 samples
    random.seed(0)
    obs_x = lea.vals(*x.random(1000000))
    # initial model, with estimated parameters
    #a0 = lea.pmf({-1: 0.1, 0: 0.1, 1: 0.5, 2: 0.2, 3: 0.1})
    #b0 = lea.pmf({-1: 0.1, 0: 0.5, 1: 0.2, 2: 0.2})
    a0 = lea.pmf({0: 0.25, 2: 0.35, 4: 0.4})
    b0 = lea.pmf({0: 0.25, 1: 0.3, 2: 0.15, 3: 0.3})
    x0 = a0 + b0
    # perform 50 steps of EM algorithm 
    model_dict = x0.learn_by_em(obs_x,nb_steps=50)
    (x1,a1,b1) = (model_dict[var] for var in (x0,a0,b0))
    assert a1.equiv_f(a,abs_tol=1e-2)
    assert b1.equiv_f(b,abs_tol=1e-2)
    assert x1.equiv_f(x,abs_tol=1e-2)
    assert isclose(obs_x.kl_divergence(x1),0.0,abs_tol=1e-5)
