Lea is a Python module aiming at working with discrete probability distributions in an intuitive way.

It allows you modeling a broad range of random phenomena: gambling, weather, finance, etc. More generally, Lea may be used for any finite set of discrete values having known probability: numbers, booleans, date/times, symbols,... Each probability distribution is modeled as a plain object, which can be named, displayed, queried or processed to produce new probability distributions.

Lea also provides advanced functions and Probabilistic Programming (PP) features; these include conditional probabilities, joint probability distributions, Bayesian networks, Markov chains and symbolic computation.

All probability calculations in Lea are performed by a new exact algorithm, the Statues algorithm, which is based on variable binding and recursive generators. For problems intractable through exact methods, Lea provides on-demand approximate algorithms, namely MC rejection sampling and likelihood weighting.

Beside the above-cited functions, Lea provides some machine learning functions, including Maximum-Likelihood and Expectation-Maximization algorithms.

Lea can be used for AI, education (probability theory & PP), generation of random samples, etc.

To install Lea, type the following command:
::

  pip install lea

Please go on Lea project page (beside) for a comprehensive documentation.