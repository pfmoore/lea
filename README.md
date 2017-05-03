# Lea - Discrete probability distributions in Python #

[comment]: <> (![Lea2_logo.png] (https://bitbucket.org/repo/BpoAoj/images/719424726-Lea2_logo.png))

**NEW**: May, 2017 - [Lea 2.3](http://pypi.python.org/pypi/lea/2.3.4) is there!
See [installation instructions](https://bitbucket.org/piedenis/lea/wiki/Installation).

## What is Lea?

Lea is a Python package aiming at working with discrete probability distributions in an intuitive way. It allows you to model a broad range of random phenomena, like dice throwing, coin tossing, gambling, weather, finance, etc. Lea lets you define random variables with given probability distributions on any set of Python objects; then, these random variables can be combined together using usual arithmetic, comparison, logical operators, as well as cartesian product, conditional probability and user-defined functions. Lea can then calculate the probability distributions on these derived random variables. Note that, by default, the _exact_ distribution is computed; a Monte-Carlo sampling method is also available, should the combinatorial becomes prohibitively large. Advanced functions let you define conditional probability tables, Bayes networks and Markov chains. With these features, Lea is definitely a toolkit for _probabilistic programming_.

Let's start by modelling a biased coin in the Python interactive console.
```
>>> from lea import *
>>> flip1 = Lea.fromValFreqs(('head',67),('tail',33))
>>> flip1
head : 67/100
tail : 33/100
```

Then let's get individual probabilities and make a random sample of 10 throws:
```
>>> P(flip1=='head')
67/100
>>> Pf(flip1=='head')
0.67
>>> flip1.random(10)
('head', 'head', 'tail', 'head', 'head', 'head', 'tail', 'head', 'tail', 'head')
```

Note that Lea internally uses integers to store probabilities, which enables for unlimited precision, even for the most unlikely event. The common usual float-point representation can be obtained when needed, using dedicated methods (see `Pf` in the example).  

We can then throw another coin, which has the same bias, and check the combinations: 
```
>>> flip2 = flip1.clone()
>>> flips = flip1 + '-' + flip2
>>> flips
head-head : 4489/10000
head-tail : 2211/10000
tail-head : 2211/10000
tail-tail : 1089/10000
>>> P(flip1==flip2)
2789/5000
>>> P(flip1!=flip2)
2211/5000
```

The following example shows how functions/methods can be applied on probability distributions and how conditional probabilities can be calculated.
```
>>> flip1.upper()
HEAD : 67/100
TAIL : 33/100
>>> flip1.upper()[0]
H : 67/100
T : 33/100
>>> def toInt(flip):
...     return 1 if flip=='head' else 0
...
>>> headCount1 = flip1.map(toInt)
>>> headCount1
0 : 33/100
1 : 67/100
>>> headCount2 = flip2.map(toInt)
>>> headCounts = headCount1 + headCount2
>>> headCounts 
0 : 1089/10000
1 : 4422/10000
2 : 4489/10000
>>> headCounts.given(flip1==flip2)
0 : 1089/5578
2 : 4489/5578
>>> headCounts.given(flip1!=flip2)
1 : 1
>>> flip1.given(headCounts==0)
tail : 1
>>> flip1.given(headCounts==1)
head : 1/2
tail : 1/2
>>> flip1.given(headCounts==2)
head : 1
```

In this example, you can notice that Lea does _lazy evaluation_, so that `flip1`, `flip2`, `headCount1`, `headCount2` and `headCounts` form a network of random variables that "remember" their causal dependencies. Based on such features, Lea can do complex probabilistic inferences like 
Markov chains and bayesian networks. For instance, the classical ["Rain-Sprinkler-Grass" bayesian network (Wikipedia)](https://en.wikipedia.org/wiki/Bayesian_network) is easily modeled in a couple of lines:
```
# -- rsg_net.py --------
from lea import *
rain = B(20,100)
sprinkler = Lea.if_(rain, B(1,100), B(40,100))
grassWet = X(sprinkler,rain).switch({ (False,False): False,
                                      (False,True ): B(80,100),
                                      (True ,False): B(90,100),
                                      (True ,True ): B(99,100) })
```
Then, this Bayesian network can be imported and queried in an interactive session:
```
>>> from rsg_net import *
>>> P(grassWet & sprinkler & rain)
99/50000
>>> Pf(grassWet & sprinkler & rain)
0.00198
>>> P(rain.given(grassWet))
891/2491
>>> Pf(rain.given(grassWet))
0.3576876756322762
>>> Pf(grassWet.given(rain))
0.8019
>>> Pf(grassWet.given(sprinkler&~rain))
0.9
>>> Pf(grassWet.given(~sprinkler&~rain))
0.0
```

The inference engine remains hidden for the user who builds and queries probabilistic models in a high-level, declarative manner. Hence, Lea is definitely a toolkit for _probabilistic programming_. Note that the Lea package includes a small _probabilistic programming language_ (PPL) called _Leapp_. It provides concise syntax to make use of Lea as easy as possible, … especially for non-Python programmers! Note that Leapp is not a true programming language. It is just a thin "syntactic sugar" layer on top of Python / Lea. The good news is that you can use standard Python syntax and put Leapp expressions as needed (or the opposite!); your favorite Python modules can be used as usual. The probabilistic programming nature of Lea/Leapp is advocated in the small apologia  [P("Hello world!") = 0.28](https://bitbucket.org/piedenis/lea/wiki/LeappPPLHelloWorld).

## Features

  * scope : finite discrete probability distributions
  * can assign probabilities on _any_ hashable Python object
  * standard distribution indicators + information theory
  * probabilities stored as integers (no rounding errors)
  * probability distribution calculus based on arithmetic, comparison, logical operators and functions
  * generation of random samples
  * joint probability tables, marginalization
  * conditional probabilities
  * Bayesian networks / inference
  * Markov chains (basic)
  * exact probabilistic inference with the "Statues algorithm", an original algorithm based on Python generators
  * approximate probabilistic inference based on Monte-Carlo sampling
  * _Leapp_, a light PPL (probabilistic programming language)
  * comprehensive tutorials (Wiki)
  * runs on Python 2 or 3
  * open-source project, LGPL license

---

# To learn more...

If you want to use Lea, see the [installation instructions](https://bitbucket.org/piedenis/lea/wiki/Installation).

[Lea basic tutorial](https://bitbucket.org/piedenis/lea/wiki/LeaPyTutorial0): if you are an absolute beginner in Lea, read this first. NO deep knowledge of probabilities nor Python is required! This tutorial exists also in [Leapp flavor](https://bitbucket.org/piedenis/lea/wiki/LeappTutorial0).

To go deeper in the tool and elaborate on probabilistic programming, there are three advanced tutorials:

  * [Lea advanced tutorial 1](https://bitbucket.org/piedenis/lea/wiki/LeaPyTutorial1) : conditional probabilities, JPD, cartesion products, …
  * [Lea advanced tutorial 2](https://bitbucket.org/piedenis/lea/wiki/LeaPyTutorial2) : CPT, bayesian networks, Markov chains
  * [Lea advanced tutorial 3](https://bitbucket.org/piedenis/lea/wiki/LeaPyTutorial3) :  MC sampling, advanced JPD, machine learning, … (new features of Lea.2.2)

The Python advanced tutorials 1 & 2 are also translated in Leapp([1](https://bitbucket.org/piedenis/lea/wiki/LeappTutorial1), [2](https://bitbucket.org/piedenis/lea/wiki/LeappTutorial2)).

[Python examples](https://bitbucket.org/piedenis/lea/wiki/Examples): to see examples of Lea in action (also translated in Leapp: [Leapp examples](https://bitbucket.org/piedenis/lea/wiki/LeappExamples)). 

If you want to understand how the Statues algorithm works, then you may have a look at [MicroLea](https://bitbucket.org/piedenis/microlea), an independent Python implementation that is much shorter and much simpler than Lea.

[Lea Wiki TOC](https://bitbucket.org/piedenis/lea/wiki): Wiki's table of contents. 

You can get also see Lea presentations made at some conferences:

* [Lea, a probability engine in Python](https://drive.google.com/open?id=0B1_ICcQCs7geUld1eE1CWGhEVEk) - presented at [FOSDEM 15/Python devroom](https://fosdem.org/2015/schedule/track/python/)
* [Probabilistic Programming with Lea](https://drive.google.com/open?id=0B1_ICcQCs7gebF9uVGdNdG1nR0E) - presented at [PyCon Ireland 15](https://python.ie/pycon-2015/)

---

# Bugs / Enhancements

If you have enhancements to propose or if you discover bugs, you are kindly invited to post issues in the present Bitbucket project page. All issues will be answered.

# Feedbacks

Don't hesitate to send your comments, questions, … by e-mail to: **pie.denis@skynet.be**, in English or French. You are welcome / _bienvenus_ !

Also, if you use Lea in your developments or researches, please tell about it! So, your experience can be shared and the project can gain recognition.

Project author, administrator: Pierre Denis — Louvain-la-Neuve, Belgium