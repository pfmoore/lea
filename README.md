# Lea - Discrete probability distributions in Python

[comment]: <> (![Lea2_logo.png] (https://bitbucket.org/repo/BpoAoj/images/719424726-Lea2_logo.png))

**NEW**: May, 2018 - [Lea 3.0.0.beta.1](http://pypi.org/project/lea/3.0.0.beta.1) is there!

## What is Lea?

Lea is a Python library aiming at working with discrete probability distributions in an intuitive way.

## Features (Lea 3)

  * **discrete probability distributions** - support: any object!
  * **random sampling**
  * **probabilisitic arithmetic**: arithmetic, comparison, logical operators and functions
  * **standard indicators** + **information theory**
  * **advanced probabilistic techniques**: Probabilisitc Programming, Bayesian reasoning, CPT, BN, JPD, MC sampling, Markov chains, …
  * **multiple probability representations**: float, decimal, fraction, …
  * **symbolic computation**, using the [SymPy library](http://www.sympy.org)
  * **exact probabilistic inference** using the _Statues_ algorithm, based on Python generators
  * **comprehensive tutorials** (Wiki)
  * **Python 2.6+ / Python 3** supported
  * **lightweight**, _pure_ Python module
  * **open-source** - LGPL license

## Some samples…

Let's start by modeling a biased coin and make a random sample of 10 throws:

[]()
```python
flip1 = lea.pmf({ 'Head': 0.75, 'Tail': 0.25 })
print(flip1)
# -> Head : 0.75
#    Tail : 0.25
print (flip1.random(10))
# -> ('Head', 'Tail', 'Tail', 'Head', 'Head', 'Head', 'Head', 'Head', 'Head', 'Head')
```

You can then throw another coin, which has the same bias, and get the probabilities of combinations: 

[]()
```python
flip2 = flip1.new()
flips = flip1 + '-' + flip2
print (flips)
# -> Head-Head : 0.5625
#    Head-Tail : 0.1875
#    Tail-Head : 0.1875
#    Tail-Tail : 0.0625
print (P(flips == 'Head-Tail'))
# -> 0.1875
print (P(flips <= 'Head-Tail'))
# -> 0.75
print (P(flip1 == flip2))
# -> 0.625
```

You can also calculate conditional probabilities, based on given information or assumptions:

[]()
```python
print (flips.given(flip2 == 'Tail'))
# -> Head-Tail : 0.75
#    Tail-Tail : 0.25
print (P((flips == 'Tail-Tail').given(flip2 == 'Tail')))
# -> 0.25
print (flip1.given(flips == 'Head-Tail'))
# -> Head : 1.0
```
With these examples, you can notice that Lea performs _lazy evaluation_, so that `flip1`, `flip2`, `flips` form a network of variables that "remember" their causal dependencies. Based on such feature, Lea can do advanced probabilistic inference like Bayesian reasoning. For instance, the classical ["Rain-Sprinkler-Grass" Bayesian network (Wikipedia)](https://en.wikipedia.org/wiki/Bayesian_network) is easily modeled in a couple of lines:

[]()
```python
rain = lea.event(0.20)
sprinkler = lea.if_(rain, lea.event(0.01),
                          lea.event(0.40))
grassWet = lea.joint(sprinkler,rain).switch({ (False,False): False,
                                              (False,True ): lea.event(0.80),
                                              (True ,False): lea.event(0.90),
                                              (True ,True ): lea.event(0.99)})
```

This BN can be queried in different ways:

[]()
```python
print (P(rain.given(grassWet)))
# -> 0.35768767563227616
print (P(grassWet.given(rain)))
# -> 0.8019000000000001
print (P(grassWet.given(sprinkler & ~rain)))
# -> 0.9000000000000001
print (P(grassWet.given(~sprinkler & ~rain)))
# -> 0.0
```

The floating-point number type is a standard although limited way to represent probabilities. Lea 3 proposes alternative representations, which can be more expressive for some domain and which are very easy to set. For example, you could use fractions: 

[]()
```python
flip1_fractions = lea.pmf({ 'Head': '75/100', 'Tail': '25/100' })
flip2_fractions = flip1_fractions.new()
flips_fractions = flip1_fractions + '-' + flip2_fractions
print (flips_fractions)
# -> Head-Head : 9/16
#    Head-Tail : 3/16
#    Tail-Head : 3/16
#    Tail-Tail : 1/16
```

You could also put variable names, which enables _symbolic computation_ of probabilities (requires [the SymPy library](http://www.sympy.org)):

[]()
```python
flip1_symbolic = lea.pmf({ 'Head': 'p', 'Tail': None })
flip2_symbolic = lea.pmf({ 'Head': 'q', 'Tail': None })
print (flip1_symbolic)
# -> Head : p
#    Tail : -p + 1
flips_symbolic = flip1_symbolic + '-' + flip2_symbolic
print (flips_symbolic)
# -> Head-Head : p*q
#    Head-Tail : -p*(q - 1)
#    Tail-Head : -q*(p - 1)
#    Tail-Tail : (p - 1)*(q - 1)
```
---

# To learn more...

The above examples show only a very, very small subset of Lea 3 capabilities. To learn more, you can read:

  * [Lea 3 Tutorial [1/3]](Lea3_Tutorial_1) - basics: building/displaying pmf, arithmetic, random sampling, conditional probabilities, …
  * [Lea 3 Tutorial [2/3]](Lea3_Tutorial_2) - standard distributions, joint distributions, Bayesian networks, Markov chains, changing probability representation, …
  * [Lea 3 Tutorial [3/3]](Lea3_Tutorial_3) - plotting, drawing without replacement, machine learning, information theory, MC estimation, symbolic computation, …
  * [Lea 3 Examples](Lea3_Examples)

Note that Lea 2 tutorials are still available [here](Home) although these are no longer maintained. You can also get Lea 2 presentation materials (note however that the syntax of Lea 3 is _not backward compatible_):

* [Lea, a probability engine in Python](https://drive.google.com/open?id=0B1_ICcQCs7geUld1eE1CWGhEVEk) - presented at [FOSDEM 15/Python devroom](https://fosdem.org/2015/schedule/track/python/)
* [Probabilistic Programming with Lea](https://drive.google.com/open?id=0B1_ICcQCs7gebF9uVGdNdG1nR0E) - presented at [PyCon Ireland 15](https://python.ie/pycon-2015/)

## On the algorithm …

The very beating heart of Lea resides in the _Statues_ algorithm, which is used for almost all probability calculations. If you want to understand how this algorithm works, then you may read [the short introduction](Lea3_Tutorial_3#markdown-header-the-statues-algorithm) or have a look at [MicroLea](https://bitbucket.org/piedenis/microlea), an independent Python implementation that is much shorter and much simpler than Lea. For a more academic description, the paper "Probabilistic Inference Using Generators" (-LINK TO ADD!-) presents the algorithm in a general and language-independent manner.

---

# Bugs / enhancements / feedback / references …

If you have enhancements to propose or if you discover bugs, you are kindly invited to [create an issue on bitbucket Lea page](https://bitbucket.org/piedenis/lea/issues). All issues will be answered!

Don't hesitate to send your comments, questions, … to [pie.denis@skynet.be](mailto:pie.denis@skynet.be), in English or French. You are welcome / _bienvenus_ !

Also, if you use Lea in your developments or researches, please tell about it! So, your experience can be shared and the project can gain recognition. Thanks!
