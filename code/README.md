# Organization

The code files are orgnized as follows:

- `comparisons.py` is the **main** file. It illustrates the behavior of the
elastic-net, LASSO and sparse envelope in a situation where there are groups of 
highly correlated variables,
- `breakpoints.py` illustrates how we can split a linear piecewise function with $2$ breakpoints into
$2$ linear piecewise functions with a single breakpoint, it can be used in the random search algorithm
but is used here only to produce the image in the beamer presentation,
- `prox_computation.py` contains how we actually compute the sparse envelope method *ie*
the FISTA algorithm and the formula for the proximal operator. It is followed by an example with
an unfit parameter to see its behavior,
- `Rando_search.py` is made of the function to compute the $0$-norm of a function,
the bisector method, an implementation of the random search algorithm and the functions needed
to compute the sparse envelope (but still secondary). There is also a test checking that the random
search algorithm works.
- `utils.py` only contains a function to save the figures without any pain in the other files.

## Improvements

The code can still be improved in the following ways:

- incorporate the random search algorithm in the proximal operator computation,
it could lead to several order of magnitude in gained time,
- write this method in a class the `scikit-learn` way with a fit, predict, and a score method
at the very least (we could also make a `GridSearchCV` for the parameters $k$ and lambda),
- take into account the sparsity of the signal with sparse matrices.