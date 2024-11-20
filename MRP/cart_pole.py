import math
from typing import Tuple

import numpy as np

from MRP.mrp import MRP

class CartPoleEnvironment(MRP):
    """Credit : https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L7"""

    def __init__(self, bins_per_feature: int = 2):

        #self.gravity = 9.8
        self.gravity = np.random.uniform(low=7, high=12) # gravity is uniformly distributed
        #self.masscart = 1.0 # masscart is uniformly distributed
        self.masscart = np.random.uniform(low=0.5, high=1.5) # masscart is uniformly distributed
        #self.masspole = 0.1 # masspole is uniformly distributed
        self.masspole = np.random.uniform(low=0.05, high=0.15) # masspole is uniformly distributed
        self.total_mass = (self.masspole + self.masscart)
        self.length = np.random.uniform(low=0.5, high=1.5)  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        #self.force_mag = 10.0
        self.force_mag = np.random.uniform(low=5, high=15) # force_mag is uniformly distributed
        #self.tau = 0.02  # seconds between state updates
        self.tau = np.random.uniform(low=0.01, high=0.05) # tau is uniformly distributed
        self.kinematics_integrator = 'euler'
        self.epsilon = np.random.rand() # epsilon controls the action distribution

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        #print('Theta threshold:', self.theta_threshold_radians) 
        # Position at which to fail the episode
        self.x_threshold = 2.4

        # Number of bins per dimension
        self.s_bins = bins_per_feature
        # Define the observation boundaries for each state variable
        # state = (x, x_dot, theta, theta_dot)
        self.obs_bounds = [[-self.x_threshold * 1.25, self.x_threshold * 1.25],
                      [-2.5, 2.5],
                      [-self.theta_threshold_radians *1.25, self.theta_threshold_radians *1.25],
                      [-2.5, 2.5]]
        # Create bins for each dimension
        self.bins = [
            np.linspace(low, high, self.s_bins + 1)[1:-1]  # Exclude the first and last bin edges
            for low, high in self.obs_bounds
        ]
        #import pdb; pdb.set_trace()
        self.total_states = self.s_bins ** 4

        # Action space
        self._all_actions = [0, 1] # left, right

        self.rewards = np.random.uniform(low=-1, high=1, size=self.total_states)

    def is_state_valid(self, state):
        x, _, theta, _ = state
        # Velocities aren't bounded, therefore cannot be checked.
        is_state_invalid = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        return not is_state_invalid
    
    def reset(self):
        """Get a random starting position."""
        state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return state

    def step(self, state):
        x, x_dot, theta, theta_dot = state
        action = np.random.binomial(1, self.epsilon) #action is random with probability epsilon
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        next_state = (x, x_dot, theta, theta_dot)

        # if we fall outside of the range, give a reward of -1 and then reset
        if not self.is_state_valid(next_state):
            next_state = self.reset()

        discretized_state_idx = self.get_discretized_feature_idx(next_state)  
        reward = self.rewards[discretized_state_idx]

        return next_state, reward

    def discretize_state(self, state: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """Discretize the continuous state into bins for each dimension."""
        discretized_state = []
        for i, s in enumerate(state):
            # For each dimension, find which bin s belongs to
            bin_indices = np.digitize(s, self.bins[i])
            discretized_state.append(bin_indices)
        return discretized_state
    
    def get_discretized_feature_idx(self, state: Tuple[float, float, float, float]) -> int:
        """Get the state index for the state."""
        discretized_state = self.discretize_state(state)
        # Calculate a unique index for the discretized state
        feature_index = 0
        num_bins = self.s_bins
        for i, bin_idx in enumerate(discretized_state):
            feature_index *= num_bins
            feature_index += bin_idx
        #import pdb; pdb.set_trace()
        return feature_index