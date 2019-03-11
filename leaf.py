'''
--------------------------------------------------------------------------------

    leaf.py

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

from .lea import Alea

def die(nb_faces=6,prob_type=None):
    ''' returns an Alea instance representing the value obtained by throwing
        a fair die with faces marked from 1 to nb_faces
    '''
    return Alea.interval(1,nb_faces,prob_type=prob_type)

def dice(nb_dice,nb_faces=6,prob_type=None):
    ''' returns an Alea instance representing the total value obtained by
        throwing nb_dice independent fair dice with faces marked from 1 to nb_faces
    '''
    return die(nb_faces,prob_type).times(nb_dice)

def dice_seq(nb_dice,nb_faces=6,sorted=True,prob_type=None):
    ''' returns an Alea instance representing the individual results obtained by
        throwing nb_dice independent fair dice with faces marked from 1 to nb_faces
        (each value is a tuple with nb_dice elements)
        * if sorted is True (default), then the combinations of dice which are
          the same apart from order are considered equal; the particular value
          used is chosen to be in order from smallest to largest value
        * if sorted is False, then all nb_dice**nb_faces combinations are produced,
          with equal probabilities
        requires Python 2.7+ for the case sorted = True
    '''
    ## for sorted = True, note that the following expression
    ##    die(nb_faces).times_tuple(nb_dice).map(lambda v: tuple(sorted(v))).get_alea()
    ## gives same results but is ineffecient as nb_dice and nb_faces grows
    ## the fast function used here is due to Paul Moore
    return die(nb_faces,prob_type).draw(nb_dice,sorted=sorted,replacement=True)


# D6 represents the value obtained by throwing a fair die with 6 faces
D6 = die(6,prob_type='r')

# flip represents a True/False boolean variable with uniform probabilities
flip = Alea.event('1/2',prob_type='r')

# card_suite is a random one-character symbol representing a card suite among
# Spades, Hearts, Diamonds and Clubs
card_suite = Alea.vals(*'SHDC',prob_type='r')

# card_rank is a one character-symbol representing a card rank among
# Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen and King
card_rank = Alea.vals(*'A23456789TJQK',prob_type='r')

# card is a random two characters symbol representing a card having a rank
# and a suite chosen in a standard deck of 52 cards
card = card_rank + card_suite
