'''
--------------------------------------------------------------------------------

    leaf.py

--------------------------------------------------------------------------------
Copyright 2013-2016 Pierre Denis

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

from .lea import *

def die(nbFaces):
    ''' returns an Alea instance representing the value obtained by throwing
        a fair die with faces marked from 1 to nbFaces
    '''
    return Lea.interval(1,nbFaces)

def dice(nbDice,nbFaces):
    ''' returns an Alea instance representing the total value obtained by
        throwing nbDice independent fair dice with faces marked from 1 to nbFaces
    '''
    return die(nbFaces).times(nbDice)

def diceSeq(nbDice,nbFaces,sorted=True):
    ''' returns an Alea instance representing the individual results obtained by
        throwing nbDice independent fair dice with faces marked from 1 to nbFaces
        (each value is a tuple with nbDice elements)
        * if sorted is True (default), then the combinations of dice which are
          the same apart from order are considered equal; the particular value
          used is chosen to be in order from smallest to largest value
        * if sorted is False, then all nbDice**nbFaces combinations are produced,
          with equal probabilities
    '''
    ## for sorted = True, note that the following expression
    ##    die(nbFaces).timesTuple(nbDice).map(lambda v: tuple(sorted(v))).getAlea()
    ## gives same results but is ineffecient as nbDice and nbFaces grows
    ## the fast function used here is due to Paul Moore
    return die(nbFaces).draw(nbDice,sorted=sorted,replacement=True)


# D6 represents the value obtained by throwing a fair die with 6 faces
D6 = die(6)

# flip represents a True/False boolean variable with uniform probabilities
flip = Lea.boolProb(1,2)

# cardSuite is a random one character symbol representing a card suite among
# Spades, Hearts, Diamonds and Clubs
cardSuite = Lea.fromSeq('SHDC')

# cardRank is a one character symbol representing a one character symbol a
# card ranks among Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen and King
cardRank = Lea.fromSeq('A23456789TJQK')

# card is a random two characters symbol representing a card having a rank
# and a suite chosen in a standard deck of 52 cards
card = cardRank + cardSuite
