# Meeting Notes

## 14/08/2026

🛠️ **First thoughts**

- don't use the term solver to describe a time-integration methods => confusion with the "internal" solver used for implicit schemes, that solves $x - \alpha f(x, t) = \beta$ for a given $(\alpha, \beta)$. Uses the term **time-stepper** instead (_e.g_ `TimeStepper` for a class, etc ...)
- what is `P` here ? and why do you need a callable for `lam` ? in general, try to avoid single letters in variable name, makes your code easier to understand ...
```python
class Dahlquist():
    # ...

    def f(self, t, u, P):
        """ Evaluate the right-hand side of the Dahlquist problem at time t. Returns the value of u'."""
        if P < 1:
            raise ValueError("P must be greater than or equal to 1.")
        """Evaluate the right-hand side of the Dahlquist problem at time t. Returns the value of u'."""
        return self.lam(P) * u + np.sin(self.alpha * t)
```
- try to be consistent with your code formatting, _e.g_ here for the exception check : either you do it on one line, or to use a new line + indent, but not both
```python
# in batpint.solver.backwardEuler.BackwardEuler
if T<t0: raise ValueError("T must be bigger than t0")
if dt <= 0:
    raise ValueError("dt must be positive")
```