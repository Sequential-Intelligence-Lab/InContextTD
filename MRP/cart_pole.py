import math
from typing import Tuple

import numpy as np

from MRP.mrp import MRP
from utils import compute_steady_dist


def tile_coding(state: np.ndarray):
    feature = np.sign(state)
    binary = (feature > 0).astype(int)
    return binary


class CartPoleEnvironment(MRP):
    """Credit : https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L7"""

    def __init__(self):

        # self.gravity = 9.8
        # gravity is uniformly distributed
        self.gravity = np.random.uniform(low=7, high=12)
        # masscart is uniformly distributed
        self.masscart = np.random.uniform(low=0.5, high=1.5)
        # masspole is uniformly distributed
        self.masspole = np.random.uniform(low=0.1, high=0.3)
        self.total_mass = (self.masspole + self.masscart)
        # actually half the pole's length
        self.length = np.random.uniform(low=0.25, high=.75)
        self.polemass_length = (self.masspole * self.length)
        # force_mag is uniformly distributed
        self.force_mag = np.random.uniform(low=5, high=15)
        # seconds between state updates
        self.tau = np.random.uniform(low=0.01, high=0.05)
        self.kinematics_integrator = 'euler'
        self.epsilon = np.random.rand()  # epsilon controls the action distribution
        # sample a random number to center the rewards at
        self.reward_center = np.random.uniform(low=0.5, high=1.5)
        self.reward_func = np.random.uniform(low=-1.0, high=1.0,
                                             size=(2, 2, 2, 2))  # sample a random reward vector

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # Position at which to fail the episode
        self.x_threshold = 2.4
        # Action space
        self._all_actions = [0, 1]  # left, right

    def is_state_valid(self, state):
        x, _, theta, _ = state
        # Velocities aren't bounded, therefore cannot be checked.
        is_state_invalid = bool(
            x < -4.8
            or x > 4.8
            or theta < -0.418
            or theta > 0.418
        )
        return not is_state_invalid

    def reset(self):
        """Get a random starting position."""
        state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return tile_coding(state)

    def step(self, state):
        x, x_dot, theta, theta_dot = state
        # action is random with probability epsilon
        action = np.random.binomial(1, self.epsilon)
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot **
                2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length *
                                                                  (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
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
            # reward = -1* np.random.uniform(low=self.reward_center-0.1, high=self.reward_center+0.1)
            next_state = self.reset()
            # print('Resetting')
        # else:
            # reward = np.random.uniform(low=self.reward_center-0.1, high=self.reward_center+0.1)
            # reward = self.reward_center
        current_feat = tile_coding(state)
        next_feat = tile_coding(next_state)
        reward = self.reward_func[*current_feat]

        return next_feat, reward
