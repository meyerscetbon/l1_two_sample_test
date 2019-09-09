# Comparing distributions: l1 geometry improves kernel two-sample testing

Code of the paper by Meyer Scetbon and Gaël Varoquaux, NeurIPS 2019

## Intro: An L1 two-sample test

We consider two sample tests: given two samples independently and identically distributed (i.i.d.) according to two probability measures P and Q, the goal of a two-sample test is to decide whether P is different from Q on the basis of the samples. We take advantage of the L1 geometry and derive a finite dimensional approximation of a particular metric which captures differences between distributions at random locations. The locations can be chosen to maximize a lower bound on test power for a statistical test using these locations.

This repository contains a Python implementation of the L1-based tests presented in our paper.

![figure](informative.png)
