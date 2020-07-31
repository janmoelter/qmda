
# Quantum Mechanical Data Assimilation

Python code implementing the Quantum Mechanical Data Assimilation (QMDA) approach introduced in

D. Giannakis. "Quantum mechanics and data assimilation". *Phys. Rev. E* **100** (2019). DOI: [10.1103/PhysRevE.100.032207](https://doi.org/10.1103/PhysRevE.100.032207).

For details and a discussion about this approach see the original publication.

## Overview

This code tries to implement an approach for sequential data assimilation introduced by [D. Giannakis](https://cims.nyu.edu/~dimitris/) using Koopman operators in Python as opposed to MATLAB, which has been used in the original publication mentioned above. Using this implementation, we reproduce some of the results presented there.

## Code

We test our implementation on the Lorenz 63 ("L63") dynamical system. The main functions to generate the data and compute the main objects are contained in two Python modules, whereas the main code for the data assimilation is contained in the Jupyter notebooks.

In our code, Python 3's ability to handle Unicode characters allows us to use Greek letters in our variable names.

## References

D. Giannakis. "Quantum mechanics and data assimilation". *Phys. Rev. E* **100** (2019). DOI: [10.1103/PhysRevE.100.032207](https://doi.org/10.1103/PhysRevE.100.032207).

D. Giannakis, S. Das, and J. Slawinska. "Reproducing kernel Hilbert space compactification of unitary evolution groups" (2018). arXiv: 1808.01515 [math.DS].

T. Berry, D. Giannakis, and J. Harlim. "Nonparametric forecasting of low-dimensional dynamical systems". *Phys. Rev. E* **91** (2015). DOI: [10.1103/PhysRevE.91.032915](https://doi.org/10.1103/PhysRevE.91.032915).
