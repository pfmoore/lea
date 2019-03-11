'''
--------------------------------------------------------------------------------

    __init__.py

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

from .lea import Lea, Alea, P, Pf
from .license import VER as __version__

# make convenient aliases for public static methods of Lea & Alea classes
bernoulli = Alea.bernoulli
binom = Alea.binom
coerce = Alea.coerce
cpt = Lea.cpt
dist_l1 = Lea.dist_l1
dist_l2 = Lea.dist_l2
event = Alea.event
func_wrapper = Lea.func_wrapper
if_ = Lea.if_
interval = Alea.interval
joint = Lea.joint
lr = Lea.lr
mutual_information = Lea.mutual_information
joint_entropy = Lea.joint_entropy
make_vars = Lea.make_vars
max_of = Lea.max_of
min_of = Lea.min_of
pmf = Alea.pmf
poisson = Alea.poisson
read_csv_file = Alea.read_csv_file
read_pandas_df = Alea.read_pandas_df
reduce_all = Lea.reduce_all
set_prob_type = Alea.set_prob_type
vals = Alea.vals
