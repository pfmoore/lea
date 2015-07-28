# Lea - Discrete probability distributions in Python #

![Lea2_logo.png] (https://bitbucket.org/repo/BpoAoj/images/719424726-Lea2_logo.png)

_Welcome in Lea!_

---

**NEW**: 17 July 2015 - [Lea 2.1.2](https://pypi.python.org/pypi/lea) is there (bug fix of `withProb` method)!

See a [presentation of Lea (pdf)](http://lea.googlecode.com/hg/images/Lea_FOSDEM15.pdf) done at  [FOSDEM 15/Python devroom](https://fosdem.org/2015/schedule/track/python/) on 31st January 2015

---

## What is Lea?

Lea is a Python package aiming at working with discrete probability distributions in an intuitive way. It allows you to model a broad range of random phenomenons, like dice throwing, coin tossing, gambling, weather, finance, etc. More generally, Lea may be used for any finite set of discrete values having known probability: numbers, booleans, date/times, symbols, … Each probability distribution is modeled as a plain object, which can be named, displayed, queried or processed to produce new probability distributions.

Lea can be used in any Python program or interactively, in the Python console. As a basic example, the statements below define and display the probability of a fair six-sided die :

```
>>> die = Lea.fromVals(1,2,3,4,5,6)
>>> die
1 : 1/6
2 : 1/6
3 : 1/6
4 : 1/6
5 : 1/6
6 : 1/6
```

and here is how you can model a biased coin, then make a random sample of 10 throws :

```
>>> flip = Lea.fromValFreqsDict({'head': 67, 'tail': 33})
>>> flip 
head : 67/100
tail : 33/100
>>> flip.random(10)
('head', 'head', 'tail', 'head', 'head', 'head', 'head', 'tail', 'head', 'head')
```

Lea allows you to compute new probability distributions from existing ones, by using different transformation means: arithmetic operators, logical operators, functions, conditions and cartesian product. For example, this is how you can count the probability distribution of number of 'heads' on 2 throws :

```
>>> flipCount = Lea.fromValFreqsDict({1: 67, 0: 33})
>>> flipCount
0 : 33/100
1 : 67/100
>>> flipCount2 = flipCount + flipCount.clone()
>>> flipCount2
0 : 1089/10000
1 : 4422/10000
2 : 4489/10000
```
then, on 4 throws :
```
>>> flipCount4 = flipCount.times(4)
0 :  1185921/100000000
1 :  9631116/100000000
2 : 29331126/100000000
3 : 39700716/100000000
4 : 20151121/100000000
>>> print (flipCount4.asFloat())
0 : 0.011859
1 : 0.096311
2 : 0.293311
3 : 0.397007
4 : 0.201511
```

Also, comparison operators can be used to derive boolean probability distributions:

```
>>> flipCount4 >= 3
False : 40148163/100000000
 True : 59851837/100000000
>>> (1 <= flipCount4) & (flipCount4 < 4)
False : 10668521/50000000
 True : 39331479/50000000
```

Lea provides a large set of operations that allow you to model complex stochastic processes. Advanced operations allow to handle conditional probabilities, including Bayesian networks and Markov chains.

## Leapp : a probabilistic programming language

As of version 2, the Lea package includes a small probabilistic programming language (PPL) called _Leapp_. It provides concise syntax to make use of Lea as easy as possible, … especially for non-Python programmers! Here are some above Python statements revamped in Leapp (`lea>` is the prompt!):

**`[`Leapp`]`**
```
lea> die = ?(1,2,3,4,5,6)
lea> flip = ?{'head': 67%, 'tail': 33%}
lea> flip 
head : 67/100
tail : 33/100
lea> flip$(10)
('head', 'head', 'head', 'head', 'head', 'tail', 'head', 'head', 'tail', 'tail')
lea> flipCount = ?{1: 67%, 0: 33%}
lea> flipCount2 = flipCount + ?flipCount
lea> flipCount4 = ?[4]flipCount
lea> :. flipCount4 
0 : 0.011859
1 : 0.096311
2 : 0.293311
3 : 0.397007
4 : 0.201511
```

Do you notice the syntax _leap_ compared to Python statements?

Note that Leapp is not a true programming language. It is just a thin "syntactic sugar" layer on top of Python / Lea. The good news is that you can use standard Python syntax and put Leapp expressions as needed (or the opposite!) ; your favorite Python modules can be used as usual.

## Lea features

Here are the main features of Lea :

  * scope : finite discrete probability distributions
  * can assign probabilities on _any_ hashable Python object
  * standard distribution indicators + information theory
  * probability stored as integers (no rounding errors)
  * probability distribution calculus based on arithmetic, comparison, boolean operators and functions
  * generation of random values
  * joint tables, marginalisation
  * conditional probabilities
  * Bayesian networks
  * Markov chains (basic)
  * _Leapp_, a light PPL (probabilistic programming language) <sup>(</sup>`*`<sup>)</sup>
  * comprehensive tutorials (Wiki)
  * runs on Python 2 or 3
  * open-source project, LGPL license

<sup>(</sup>`*`<sup>)</sup> The "_probabilistic programming_" nature of Lea/Leapp is advocated in the small apologia  [P("Hello world!") = 0.28](LeappPPLHelloWorld.md).

---


To learn more, read the [Lea tutorial - Leapp flavor](LeappTutorial.md) or the [Lea tutorial - Python flavor](LeaPyTutorial.md).

For installation instructions, see [Installation](Installation.md).


---


Please send your comments, critics, suggestions, bug reports, … in English or French by E-mail to: **pie.denis@skynet.be**. You are more than welcome / _bienvenus_ !

You can also post issues in the present project site.

Project author, administrator : Pierre Denis (Louvain-la-Neuve, Belgium)