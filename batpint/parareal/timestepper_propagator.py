from batpint.parareal.base_propagator import Propagator    
from batpint.parareal.propagation_state import PropagationState
class TimeStepperPropagator(Propagator):
    """
    Propagator based on an event-driven TimeStepper.
    """

    def __init__(self, timestepper, direction=0, terminate_cycle=None):
        self.timestepper = timestepper
        self.direction = direction
        self.terminate_cycle = terminate_cycle if terminate_cycle is not None else lambda state: False
        self.history = self.timestepper.history if self.timestepper.save_history else {"t": [], "u": []}
        self.current_cycle = 0
        
    @property
    def state(self):
        return self.timestepper.state
    
    def make_state(self, t, u, cycle):
        return PropagationState(t, u, cycle, u_event=u)
    
    def propagate(self, t, u, cycle): 
        state = self.make_state(t, u, cycle)
        self.timestepper.set_state(state)
        if self.terminate_cycle(self.timestepper.state):
            raise RuntimeError(
            f"CYCLE_TERMINATION: cycle={self.timestepper.state.cycle}, "
            f"t={self.timestepper.state.t}, "
            f"u={self.timestepper.state.u}"
                )
        self.timestepper.advance_to_event(direction=self.direction)
        return self.timestepper.state.t, self.timestepper.state.u

        