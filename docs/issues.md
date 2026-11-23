# Issues detected when investigating the code

## 1 : script `07_pybammEvents.py`

1. `solutionFine` contains both the solution and derivative, even if the derivative is not needed ...

## Generic advices

- try not go over 80 characters per line of code, including comments (or rarely when it barely goes over the limit)
- no need of `copy.deepcopy` with numpy array : they have already a `copy()` method that should be used instead
