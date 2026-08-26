from dataclasses import dataclass
import inspect
from distro import name
import numpy as np


class Problem:
    """
    The problem (IVP) is defined by
        du/dt = f(t, u), u(t0) = u0.
    Optional functions may be supplied for the Jacobian of the
    right-hand side and for event detection.
    Additional keyword arguments passed at construction are stored
    as fixed problem parameters.
    
    Parameters
    ----------
    t0 : float
        Initial time.
    u0 : scalar or array-like
        Initial state.
    rhs : callable
        Right-hand side with convention f(t, u, **kwargs).
    jacobian : callable, optional
        Jacobian with convention J(t, u, **kwargs).
    event : callable, optional
        Event function with convention g(t, u, **kwargs).
    **params
        Fixed parameters of the mathematical problem.
    """

    def __init__(self, t0, u0, rhs, jacobian=None, event=None, terminate=None, **params):
        # Original initial condition of the IVP
        self.t_start = t0
        self.u_start = u0
        
        self.rhs = rhs
        self.jacobian = jacobian
        self.event = event
        self.terminate = terminate
        self.params = params

        self.rhs_params = self._get_params(rhs)
        self.jacobian_params = self._get_params(jacobian)
        self.event_params = self._get_params(event)
        self.terminate_params = self._get_params(terminate)

    @staticmethod
    def _get_params(func):
        if func is None:return ()
        names = tuple(inspect.signature(func).parameters)
        if names[:2] != ("t", "u"):
            raise ValueError("Problem functions must start with arguments (t, u).")

        return names[2:]
    
    def __call__(self, t, u, current_params=None):
        """
        Evaluate the right-hand side f(t, u).
        """
        if current_params is None:
            current_params = self.params

        kwargs = {name: getattr(current_params, name) for name in self.rhs_params}
        return self.rhs(t, u, **kwargs)

    def jacobian_value(self, t, u, current_params=None):
        """
        Evaluate the Jacobian of the right-hand side.
        """
        if self.jacobian is None: 
            raise ValueError("No Jacobian function defined for this problem.")
        if current_params is None:
                    current_params = self.params

        kwargs = {name: getattr(current_params, name) for name in self.jacobian_params}
        return self.jacobian(t, u, **kwargs)

    def event_value(self, t, u, current_params=None):
        """
        Evaluate the event function g(t, u).
        """
        if self.event is None: 
            raise ValueError("No event function defined for this problem.")
        if current_params is None:
                    current_params = self.params

        kwargs = {name: getattr(current_params, name) for name in self.event_params}
        return self.event(t, u, **kwargs)

    def termination_value(self, t, u, current_params=None):
        """
        Evaluate the termination function.
        """
        if self.terminate is None: 
            return False  # No termination function defined
        if current_params is None:
                    current_params = self.params

        kwargs = {name: getattr(current_params, name) for name in self.terminate_params}
        return self.terminate(t, u, **kwargs)


class Parameters:
    """
    Parameters of the Problem class
    Attributes
    ----------

    """
    def __init__(self, *parameters):
        self._parameters = {}

        for parameter in parameters:
            self.add(parameter)

    def add(self, parameter):
        if not isinstance(parameter, Parameter):
            raise TypeError("parameter must be an instance of Parameter class")
        if parameter.name in self._parameters:
            raise ValueError(f"Parameter '{parameter.name}' already exists.")

        self._parameters[parameter.name] = parameter

    def __contains__(self, name):
        return name in self._parameters

    def __getitem__(self, name):
        return self._parameters[name]


class Parameter:
    """
    Definition of a single problem parameter.

    A parameter is either:
    - stored: defined by a value
    - functional: defined by a function of (t, u, ...)

    Stored parameters may optionally define an update rule.
    Functional parameters remain functions and cannot be changed.
    """

    def __init__(self, name, value=None, function=None, update=None):
        self.name = name
        self._value = value
        self.function = function
        self.update = update

        self.dependencies = ()
        self._set_dependencies()

    @property
    def value(self):
        if self.function is not None:
            raise ValueError("Cannot get value of a functional parameter.")
        return self._value
    
    def _set_dependencies(self):
        if self.function is not None:
            sig = inspect.signature(self.function)
            self.dependencies = tuple(name for name in sig.parameters if name not in ("t", "u"))

class CurrentParameters:
    def __init__(self, base_params):
        self.base = base_params
        self._values = {}

    def get(self, name, t=None, u=None):
        # Updated/current stored value
        if name in self._values:
            return self._values[name]

        parameter = self.base[name]

        # Stored parameter
        if parameter.function is None:
            return parameter.value

        # Functional parameter
        kwargs = {
            dep: self.get(dep, t, u)
            for dep in parameter.dependencies
        }

        return parameter.function(t, u, **kwargs)

    def set(self, name, value):
        self._values[name] = value