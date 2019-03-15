'''
--------------------------------------------------------------------------------

    blea.py

--------------------------------------------------------------------------------
Copyright 2013-2019 Pierre Denis

This file is part of Lea.

Lea is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Lea is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Lea.  If not, see <http://www.gnu.org/licenses/>.
--------------------------------------------------------------------------------
'''

from .lea import Lea
from .alea import Alea
from .clea import Clea
from .ilea import Ilea
from .prob_fraction import ProbFraction
from .toolbox import dict, zip
from operator import or_
from itertools import chain
            
class Blea(Lea):
    
    '''
    Blea is a Lea subclass, which instance represents a conditional probability
    table (CPT), as a set of Ilea instances. Each Ilea instance represents a
    given distibution <Vi,p(Vi|C)>, assuming a given condition C is verified,
    in the sense of a conditional probability.
    The set of conditions shall form a partition of the "certain true", i.e.
     ORing  all conditions shall give a "certain true" distribution
     ANDing all conditions pairwise shall give "certain false" distributions
    A Blea instance is a "misxture pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''

    __slots__ = ('_ileas','_cond_clea')
    
    def __init__(self,*ileas,**kwargs):
        Lea.__init__(self)
        self._ileas = ileas
        # _cond_lea is used only by _gen_one_random_mc method
        self._cond_clea = None

    __arg_names_of_build_meth = frozenset(('prior_lea','auto_else','check'))

    @staticmethod
    def build(*clauses,**kwargs):
        ''' see Lea.build_cpt method        
        '''
        arg_names = frozenset(kwargs.keys())
        unknown_arg_names = arg_names - Blea.__arg_names_of_build_meth
        if len(unknown_arg_names) > 0:
            raise Lea.Error("unknown argument keyword '%s'; shall be only among %s"%(next(iter(unknown_arg_names)),tuple(Blea._arg_names_of_build_meth)))
        prior_lea = kwargs.get('prior_lea',None)
        auto_else = kwargs.get('auto_else',False)
        check = kwargs.get('check',True)
        else_clause_results = tuple(result for (cond,result) in clauses if cond is None)
        if len(else_clause_results) > 1:
            raise Lea.Error("impossible to define more than one 'else' clause")
        if len(else_clause_results) == 1:
            if prior_lea is not None:
                raise Lea.Error("impossible to define together prior probabilities and 'else' clause")
            if auto_else:
                raise Lea.Error("impossible to have auto_else=True and 'else' clause")                
            else_clause_result = else_clause_results[0]
        elif auto_else:
            if prior_lea is not None:
                raise Lea.Error("impossible to define together prior probabilities and auto_else=True")
            # take uniform distribution on all values found in clause's results (principle of indifference)
            else_clause_result = Alea.vals(*frozenset(val for (cond,result) in clauses for val in Alea.coerce(result).support))
        else:
            else_clause_result = None
        # get clause conditions and results, excepting 'else' clause, after coercion to Lea instances
        norm_clauses = ((Alea.coerce(cond),Alea.coerce(result)) for (cond,result) in clauses if cond is not None)
        (cond_leas,res_leas) = tuple(zip(*norm_clauses))
        # check that conditions are disjoint
        if check:
            clea_ = Clea(*cond_leas)
            clea_._init_calc()
            if any(v.count(True) > 1 for (v,_) in clea_.gen_vp()):
                raise Lea.Error("clause conditions are not disjoint")
        # build the OR of all given conditions, excepting 'else'
        or_conds_lea = Lea.reduce_all(or_,cond_leas,True)
        if prior_lea is not None:
            # prior distribution: determine else_clause_result
            if check and or_conds_lea.is_true():
                # TODO check prior_lea equivalent to self
                raise Lea.Error("forbidden to define prior probabilities for complete clause set")
            p_true = or_conds_lea._p(True)
            p_false = 1 - p_true
            prior_alea_dict = dict(prior_lea.get_alea()._gen_vp())
            norm_alea_dict = dict(Alea.vals(*res_leas).flat().get_alea()._gen_vp())
            values_set = frozenset(chain(prior_alea_dict.keys(),norm_alea_dict.keys()))
            vps = []
            for value in values_set:
                 prior_p = prior_alea_dict.get(value,0)
                 cond_p = norm_alea_dict.get(value,0)
                 p = prior_p - cond_p*p_true
                 if not(0 <= p <= p_false):
                     # Infeasible : probability represented by p goes outside range from 0 to 1
                     lower_p = cond_p * p_true
                     upper_p = cond_p*p_true + p_false
                     raise Lea.Error("prior probability of '%s' is %s, outside the range [ %s , %s ]"%(value,prior_p,lower_p,upper_p))
                 vps.append((value,p))
            else_clause_result = Alea.pmf(vps)
        elif else_clause_result is None:
            # no 'else' clause: check that clause set is complete
            if check and not or_conds_lea.is_true():
                raise Lea.Error("incomplete clause set requires 'else' clause or auto_else=True or prior_lea=...")
        if else_clause_result is not None:
            # add the else clause
            else_cond_lea = ~or_conds_lea
            ## other equivalent statement: else_cond_lea = Lea.reduce_all(and_,(~cond_lea for cond_lea in cond_leas))
            else_clause_result = Alea.coerce(else_clause_result)
            res_leas += (else_clause_result,)
            cond_leas += (else_cond_lea,)
            # note that or_conds_lea is NOT extended with or_conds_lea |= else_cond_lea
            # so, in case of else clause (and only in this case), or_conds_lea is NOT certainly true
        # build a Blea, providing a sequence of new Ileas for each of the clause 
        return Blea(*(Ilea(res_lea,(cond_lea,)) for (res_lea,cond_lea) in zip(res_leas,cond_leas)))

    def _get_lea_children(self):
        return self._ileas
    
    def _gen_vp(self):
        for i_lea in self._ileas:
            for vp in i_lea.gen_vp():
                yield vp

    def _gen_one_random_mc(self):
        if self._cond_clea is None:
            # _cond_alea is a cartesian product of all Alea leaves present in CPT conditions;
            cond_alea_leaves_set = frozenset(alea_leaf for ilea in self._ileas                             \
                                                   for alea_leaf in ilea._cond_leas[0].get_alea_leaves_set())
            self._cond_clea = Clea(*cond_alea_leaves_set)
        # the first for loop binds a random value on each Alea instances refered in CPT conditions
        for _ in self._cond_clea._gen_one_random_mc():
            # here, there will be at most one ilea having condition that evaluates to True,
            # regarding the random binding that has been made 
            for i_lea in self._ileas:
                for v in i_lea._gen_one_random_mc_no_exc():
                    if v is not i_lea:
                        # the current ilea is the one having the condition that evaluates to True
                        yield v
