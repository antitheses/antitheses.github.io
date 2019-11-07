---
layout: post
title: Introduction to Bayesian Analysis
---

Deductive logic, given a cause we can work out the consequences. The logic used in "pure" mathematics to derive many complicated and useful results from as a logical consequence of a few well-defined axioms.

The entirety of Linear Algebra can be derived from the following axioms:

```
...


```

Given a certain effects what are the possible causes. For this sort of investigation, we need Inductive Logic. 

- Cox's Transitive Ranking

  - Larger the numerical value assigned to a proportion, more we believe it

  - What rules must these numbers obey to ensure logical consitency?

  - If we specify how much we believe something is true, we have specified how much we believe something is false (implicitly)

  - If we specify how much we believe * is true

    - Y is true

    - X is true given that Y is true

    - (implicitly specify) both X and Y are true

      - $prob(X|I) + prob(\bar{X}|I) = 1$

      - $prob(X, Y |I) = prob(X|Y, I) \times prob(Y|I)$

      - > comma is read as conjuction 'and'

    - Our probabilities are condionals on I which denotes relevant background information at hand

      > There is no such thing as absolute probability.

    - We must never forget $I$'s existence

- Equation 1 is called the sum rule; Equation 2 is called the product rule

- 1 and 2 form the basic algebra of probability theory

- Two most useful results derived from 1 and 2 are

  - Bayes rule : $prob(X|Y, I) = \frac{prob(Y|X, I) \times prob(X|I)}{prob(Y|I)}$
  - Marginalization : $prob(X|I) = \int_{-\infty}^{+\infty} prob(X, Y|I)dY = \int_{-\infty}^{+\infty}prob(X|Y, I) \times prob(Y|I)dY$

- Bayes rule enables use to turn the symbols around w.r.t. to conditioning

- Replace X and Y with hypothesis and data

  - $prob(hypothesis|data, I)\ \alpha\ prob(data | hypothesis, I) \times prob(hypothesis | I)$ 
  - Formal names of quantities
    - $prob(hypothesis | I) \rightarrow $prior probability
    - This is modified by the experimental measurements through the likelihood function $prob(data|hypothesis, I)$

> How to reason in situations where its impossible to argue with absolute certainty?

- Frequentist definition
  - Probability : long-run relative frequency with which an event occurred given infinitely many repeated trials
- Laplace estimated the mass of saturn

